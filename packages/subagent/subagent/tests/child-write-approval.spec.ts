/**
 * The opted-in parent-GUI route for one delegated ask class: a child's
 * out-of-workspace single-file `write` escalation. Covers the default denial,
 * the explicit opt-in, one approved original call, a rejection that never writes,
 * the parent-session gates (policy and open turn), parallel isolation, the
 * fail-closed timeout, and the covered-class judge.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mkdtemp, readFile, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import SandboxedFileSystem from '@deepseek-ai/dsh-fs-sandbox'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import SandboxPolicyService from '@deepseek-ai/dsh-sandbox-policy'
import { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import SubagentRuntime from '@deepseek-ai/dsh-subagent'
import * as ToolFs from '@deepseek-ai/dsh-tool-fs'
import ApprovalService from '@deepseek-ai/dsh-user-approval'
import type { ApprovalRequestEvent } from '@deepseek-ai/dsh-user-approval/types'
import { MockAdapter, textResponse, toolCallResponse } from '../../../core/agent-loop/tests/mock-adapter.ts'
import { childWriteApprovalDisplayReason, isInsideWorkspaceRoot, readChildWriteEscalation } from '../src/child-write-approval.ts'
import { mountLocalActivations, startPreparedActivation } from './local-activation.ts'

type Script = ConstructorParameters<typeof MockAdapter>[0]

const contexts: Context[] = []
let workspace: string
let outside: string

beforeEach(async () => {
  workspace = await realpath(await mkdtemp(join(tmpdir(), 'dsh-child-write-')))
  outside = await realpath(await mkdtemp(join(tmpdir(), 'dsh-child-write-outside-')))
})

afterEach(async () => {
  for (const ctx of contexts.splice(0).reverse()) await ctx.fiber.dispose()
  await rm(workspace, { recursive: true, force: true })
  await rm(outside, { recursive: true, force: true })
})

/** Mount the production composition a delegated child's write escalation runs through. */
async function setup(script: Script, subagentConfig: Record<string, unknown> = {}): Promise<{ ctx: Context; parent: Agent }> {
  const ctx = new Context()
  contexts.push(ctx)
  await mountAgentLoopTestDependencies(ctx)
  await mountLocalActivations(ctx)
  await ctx.plugin(SubagentRuntime, subagentConfig)
  await ctx.plugin(SandboxPolicyService, { mode: 'workspace-write', workspaceRoot: workspace })
  await ctx.plugin(SandboxedFileSystem, { cwd: workspace })
  await ctx.plugin(ToolFs)
  await ctx.plugin(ApprovalService)
  await ctx.plugin(AgentLoop, { agents: [] })
  ctx.llm.registerAdapter(['mock'], new MockAdapter(script))
  const parent = await ctx.agentLoop.create(SessionId('parent'), { provider: 'mock', model: 'mock' }, { cwd: workspace })
  return { ctx, parent }
}

function spawnRequest(parent: Agent) {
  return {
    label: 'child task',
    prompt: [{ type: 'text' as const, text: 'child task' }],
    parent,
    signal: new AbortController().signal,
  }
}

function toolResultTexts(agent: Agent): string[] {
  return agent.session.snapshotEvents()
    .filter((event): event is SessionEvent<'tool/result'> => event.type === 'tool/result')
    .map(event => event.data.message.content
      .filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
      .map(block => block.text)
      .join(''))
}

function auditOf(agent: Agent): { asked: SessionEvent<'approval/asked'>[]; decided: SessionEvent<'approval/decided'>[] } {
  const events = agent.session.snapshotEvents()
  return {
    asked: events.filter((event): event is SessionEvent<'approval/asked'> => event.type === 'approval/asked'),
    decided: events.filter((event): event is SessionEvent<'approval/decided'> => event.type === 'approval/decided'),
  }
}

/** Script one escaped single-file write followed by a closing text response. */
function escalationScript(target: string, justification: string): Script {
  return [
    toolCallResponse('write-call', 'write', {
      file_path: target,
      content: 'escaped',
      sandbox_permissions: 'danger-full-access',
      justification,
    }),
    textResponse('child done'),
  ]
}

describe('child write approval forwarding', () => {
  it('denies by default: the pinned never policy rejects before any answerer', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script)
    const target = join(outside, 'default-denied.txt')
    const consulted: ApprovalRequestEvent[] = []
    ctx.on('approval/request', (request) => {
      consulted.push(request)
      return Promise.resolve('allowed-once' as const)
    })
    script.push(...escalationScript(target, 'default deny check'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      const child = run.localAgent
      expect(child.session.snapshotEvents().filter(event => event.type === 'approval/policy'))
        .toMatchObject([{ data: { policy: 'never', source: 'delegation' } }])
      expect(consulted).toHaveLength(0)
      await expect(readFile(target, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
      expect(toolResultTexts(child).join('\n'))
        .toContain('the user rejected escalating this operation to "danger-full-access"')
    } finally {
      await run.dispose()
    }
  })

  it('forwards one approved out-of-workspace write to the parent session and runs the original call once', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward' })
    const target = join(outside, 'approved.txt')
    parent.session.append('turn/start', { turn: 1 })
    const consulted: ApprovalRequestEvent[] = []
    ctx.on('approval/request', (request) => {
      consulted.push(request)
      return Promise.resolve(request.agent === parent ? 'allowed-once' as const : 'rejected' as const)
    })
    script.push(...escalationScript(target, 'publish the report outside the workspace'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      const child = run.localAgent
      expect(await readFile(target, 'utf8')).toBe('escaped')
      // The parent session's own approval audit recorded the human decision.
      const parentAudit = auditOf(parent)
      expect(parentAudit.asked).toHaveLength(1)
      expect(parentAudit.decided).toMatchObject([{ data: { outcome: 'allowed-once' } }])
      expect(consulted).toHaveLength(1)
      expect(consulted[0]!.agent).toBe(parent)
      expect(consulted[0]!.toolName).toBe('write')
      // The parent presentation names the real child session and the exact target.
      expect(consulted[0]!.displayReason?.en).toContain(child.session.id)
      expect(consulted[0]!.displayReason?.en).toContain(target)
      expect(consulted[0]!.reason).toContain(target)
      // The child's own audit pair stays the record of the original call.
      const childAudit = auditOf(child)
      expect(childAudit.asked).toMatchObject([{ data: { toolName: 'write' } }])
      expect(childAudit.decided).toMatchObject([{ data: { outcome: 'allowed-once' } }])
      expect(toolResultTexts(child).join('\n')).not.toContain('[sandbox:')
    } finally {
      await run.dispose()
    }
  })

  it('never writes when the user rejects the forwarded request', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward' })
    const target = join(outside, 'rejected.txt')
    parent.session.append('turn/start', { turn: 1 })
    ctx.on('approval/request', () => Promise.resolve('rejected' as const))
    script.push(...escalationScript(target, 'user rejects this one'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      await expect(readFile(target, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
      expect(JSON.stringify(auditOf(parent).decided)).toContain('rejected')
      expect(toolResultTexts(run.localAgent).join('\n'))
        .toContain('the user rejected escalating this operation to "danger-full-access"')
    } finally {
      await run.dispose()
    }
  })

  it('fails closed when the parent session turn has ended before the ask', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward' })
    const target = join(outside, 'no-parent-turn.txt')
    const consulted: ApprovalRequestEvent[] = []
    ctx.on('approval/request', (request) => {
      consulted.push(request)
      return Promise.resolve('allowed-once' as const)
    })
    script.push(...escalationScript(target, 'parent turn already ended'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      // The audit pair must be turn-enclosed: no open parent turn means no ask.
      expect(consulted).toHaveLength(0)
      await expect(readFile(target, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
      expect(toolResultTexts(run.localAgent).join('\n'))
        .toContain('requires approval, but no approval channel is available')
      expect(JSON.stringify(auditOf(run.localAgent).decided)).toContain('unavailable')
    } finally {
      await run.dispose()
    }
  })

  it('keeps the pinned never policy when the parent session disables approvals', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward' })
    ctx.approval.setPolicy(parent, 'never')
    const target = join(outside, 'parent-never.txt')
    const consulted: ApprovalRequestEvent[] = []
    ctx.on('approval/request', (request) => {
      consulted.push(request)
      return Promise.resolve('allowed-once' as const)
    })
    script.push(...escalationScript(target, 'parent disabled approvals'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      const child = run.localAgent
      // The parent's own policy wins at delegation: no forwarding route exists.
      expect(child.session.snapshotEvents().filter(event => event.type === 'approval/policy'))
        .toMatchObject([{ data: { policy: 'never', source: 'delegation' } }])
      expect(consulted).toHaveLength(0)
      await expect(readFile(target, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
    } finally {
      await run.dispose()
    }
  })

  it('does not forward an escalation whose target is inside the workspace', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward' })
    const target = join(workspace, 'inside.txt')
    parent.session.append('turn/start', { turn: 1 })
    const consulted: ApprovalRequestEvent[] = []
    ctx.on('approval/request', (request) => {
      consulted.push(request)
      return Promise.resolve('allowed-once' as const)
    })
    script.push(...escalationScript(target, 'redundant inside-workspace escalation'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      expect(consulted).toHaveLength(0)
      await expect(readFile(target, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
      expect(toolResultTexts(run.localAgent).join('\n'))
        .toContain('the user rejected escalating this operation to "danger-full-access"')
    } finally {
      await run.dispose()
    }
  })

  it('fails closed when the forwarded ask outlives its timeout', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward', childWriteApprovalTimeoutMs: 100 })
    const target = join(outside, 'timed-out.txt')
    parent.session.append('turn/start', { turn: 1 })
    ctx.on('approval/request', () => new Promise<never>(() => {}))
    script.push(...escalationScript(target, 'nobody answers in time'))
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      await expect(readFile(target, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
      expect(toolResultTexts(run.localAgent).join('\n'))
        .toContain('approval for escalating to "danger-full-access" was cancelled')
    } finally {
      await run.dispose()
    }
  })

  it('isolates two parallel children: one granted call does not leak to the other', async () => {
    const script: Script = []
    const { ctx, parent } = await setup(script, { childWriteApproval: 'forward' })
    const approvedTarget = join(outside, 'parallel-approved.txt')
    const deniedTarget = join(outside, 'parallel-denied.txt')
    parent.session.append('turn/start', { turn: 1 })
    const consulted: ApprovalRequestEvent[] = []
    ctx.on('approval/request', (request) => {
      consulted.push(request)
      return Promise.resolve(request.agent === parent && (request.reason ?? '').includes(approvedTarget)
        ? 'allowed-once' as const
        : 'rejected' as const)
    })
    // Each child runs its own model route so the two scripts never interleave.
    const scriptA: Script = []
    const scriptB: Script = []
    ctx.llm.registerAdapter(['mock-a'], new MockAdapter(scriptA))
    ctx.llm.registerAdapter(['mock-b'], new MockAdapter(scriptB))
    scriptA.push(...escalationScript(approvedTarget, 'parallel child one'))
    scriptB.push(...escalationScript(deniedTarget, 'parallel child two'))
    const runA = await startPreparedActivation(
      { ...spawnRequest(parent), agentOptions: { provider: 'mock-a', model: 'mock-a' } },
      {},
    )
    const runB = await startPreparedActivation(
      { ...spawnRequest(parent), agentOptions: { provider: 'mock-b', model: 'mock-b' } },
      {},
    )
    try {
      await Promise.all([runA.result, runB.result])
      expect(consulted).toHaveLength(2)
      expect(consulted.filter(request => (request.reason ?? '').includes(approvedTarget))).toHaveLength(1)
      expect(auditOf(parent).decided.map(event => event.data.outcome).sort()).toEqual(['allowed-once', 'rejected'])
      // The granted call produced one write; the rejected parallel call never wrote.
      expect(await readFile(approvedTarget, 'utf8')).toBe('escaped')
      await expect(readFile(deniedTarget, 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
    } finally {
      await Promise.all([runA.dispose(), runB.dispose()])
    }
  })

  it('judges the covered class from the recorded call arguments, not from the request text', async () => {
    const script: Script = []
    const { parent } = await setup(script, { childWriteApproval: 'forward' })
    const target = join(outside, 'judge-file.txt')
    parent.session.append('turn/start', { turn: 1 })
    const run = await startPreparedActivation(spawnRequest(parent), {})
    try {
      await run.result
      const child = run.localAgent
      const request = { agent: child, toolName: 'write', callId: 'call-1' } as unknown as ApprovalRequestEvent
      const covered = JSON.stringify({
        file_path: target,
        content: 'x',
        sandbox_permissions: 'danger-full-access',
        justification: 'needed',
      })
      expect(readChildWriteEscalation(child, request, covered)).toMatchObject({
        callId: 'call-1',
        target,
        justification: 'needed',
      })
      // A different tool, an unrecorded call, an inside-workspace target, or a
      // missing justification is never the covered class.
      expect(readChildWriteEscalation(child, { ...request, toolName: 'bash' }, covered)).toBeUndefined()
      expect(readChildWriteEscalation(child, request, undefined)).toBeUndefined()
      expect(readChildWriteEscalation(child, request, JSON.stringify({
        file_path: join(workspace, 'inside.txt'),
        content: 'x',
        sandbox_permissions: 'danger-full-access',
        justification: 'needed',
      }))).toBeUndefined()
      expect(readChildWriteEscalation(child, request, JSON.stringify({
        file_path: join(outside, 'no-justification.txt'),
        content: 'x',
        sandbox_permissions: 'danger-full-access',
      }))).toBeUndefined()
    } finally {
      await run.dispose()
    }
  })
})

describe('child write approval judge', () => {
  it('recognizes only targets outside the workspace root', () => {
    const root = join(tmpdir(), 'judge-root')
    expect(isInsideWorkspaceRoot(root, join(root, 'a.txt'))).toBe(true)
    expect(isInsideWorkspaceRoot(root, root)).toBe(true)
    expect(isInsideWorkspaceRoot(root, join(root, '..', 'escape.txt'))).toBe(false)
    expect(isInsideWorkspaceRoot(root, join(tmpdir(), 'other.txt'))).toBe(false)
  })

  it('localizes the parent presentation with the child session, target, and justification', () => {
    const reason = childWriteApprovalDisplayReason('child-7', '/tmp/report.txt', 'user needs the file')
    expect(reason.en).toContain('child-7')
    expect(reason.en).toContain('/tmp/report.txt')
    expect(reason.en).toContain('user needs the file')
    expect(reason.zh).toContain('child-7')
    expect(reason.zh).toContain('/tmp/report.txt')
    for (const text of Object.values(reason)) expect(text.length).toBeGreaterThan(0)
  })
})
