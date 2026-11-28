/**
 * Opt-in forwarding of ONE delegated ask to the parent session's user: a child's
 * out-of-workspace single-file `write` escalation. A delegated child's approval
 * policy is pinned `never`, so every ask resolves `rejected` before any answerer
 * runs; with this module installed the deployment may instead route exactly that
 * ask to the parent session's own approval channel (the GUI the user is looking
 * at) and resolve the child's original call with the user's decision.
 *
 * The forwarded ask is a real, audited parent-session approval: the parent's
 * `never` policy still rejects it before any answerer, no open parent turn
 * rejects it, and nothing here changes a sandbox mode — an approved escalation
 * stays the existing one-shot per-call grant that the child's own tool call
 * consumes. Every other ask from the child is denied here, so enabling
 * forwarding never widens the child beyond the covered class.
 *
 * Parent presentation shows the real child Session id, the original tool call's
 * target, and the model's justification through the localized `displayReason`;
 * the grant is bound to the original call id and to a digest of the logged
 * tool-call arguments, re-checked immediately before the grant is returned.
 *
 * @module @deepseek-ai/dsh-subagent/child-write-approval
 */

import { createHash } from 'node:crypto'
import { isAbsolute, relative, resolve, sep } from 'node:path'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { ToolCallId } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-user-approval'
import type { ApprovalOutcome, ApprovalRequestEvent } from '@deepseek-ai/dsh-user-approval/types'
import type {} from '@deepseek-ai/dsh-working-directory'
import { delegationDepthOf } from './depth.ts'

/** The deployment switch: `deny` keeps the pinned `never` child policy, `forward` opts into the one covered ask class. */
export type ChildWriteApprovalMode = 'deny' | 'forward'

/**
 * Settings read at ask time rather than at delegation, so switching the
 * deployment back to `deny` fails the next ask closed instead of keeping a
 * forwarding route alive for an already-running child.
 */
export interface ChildWriteApprovalSettings {
  /** Whether the covered ask class may reach the parent session's user. */
  readonly mode: ChildWriteApprovalMode
  /** Fail-closed lifetime of one forwarded ask, in milliseconds. */
  readonly timeoutMs: number
}

/**
 * Arguments recorded for one live child tool call, keyed by call id. The manager
 * feeds it from delivered `session/event` records and drops each entry at its
 * `tool/result`, so the forwarder never reads historical Session positions.
 */
export type ChildWriteCallLedger = ReadonlyMap<ToolCallId, string>

/** Default fail-closed lifetime of one forwarded ask: five minutes. */
export const DEFAULT_CHILD_WRITE_APPROVAL_TIMEOUT_MS = 300_000

/**
 * Model-facing delegation-scope sentence appended only while forwarding is in
 * effect, so the child knows which asks can reach a human and which stay denied.
 */
export const CHILD_WRITE_APPROVAL_CONTEXT
  = 'One exception applies to this delegation: a single-file `write` whose target lies outside your working '
    + 'directory may be sent to the delegating session\'s user for one-shot approval and, if approved, runs once '
    + 'with the wider permission. Every other operation that requires approval is rejected — do not retry it.'

/** One resolved, still-unconsumed escalation described by the child's own tool call. */
export interface ChildWriteEscalation {
  /** The original tool call the decision is bound to. */
  readonly callId: ToolCallId
  /** Absolute canonical-enough target the write names. */
  readonly target: string
  /** The model's one-sentence justification from the call arguments. */
  readonly justification: string
  /** SHA-256 of the logged `tool/call` arguments; re-checked before the grant is returned. */
  readonly argsDigest: string
}

/** Locale copy for the parent-session presentation. */
const LOCALES = {
  en: (child: string, target: string, justification: string): string =>
    `Allow one out-of-workspace file write by subagent ${child} to ${target}? ${justification}`,
  zh: (child: string, target: string, justification: string): string =>
    `是否允许子代理 ${child} 在工作区外写入单个文件 ${target}？${justification}`,
  ja: (child: string, target: string, justification: string): string =>
    `サブエージェント ${child} がワークスペース外のファイル ${target} へ 1 回だけ書き込むことを許可しますか？${justification}`,
  es: (child: string, target: string, justification: string): string =>
    `¿Permitir una escritura fuera del espacio de trabajo del subagente ${child} en ${target}? ${justification}`,
  fr: (child: string, target: string, justification: string): string =>
    `Autoriser une écriture hors espace de travail du sous-agent ${child} vers ${target} ? ${justification}`,
  de: (child: string, target: string, justification: string): string =>
    `Eine Datei außerhalb des Arbeitsbereichs durch Subagent ${child} nach ${target} schreiben? ${justification}`,
  ru: (child: string, target: string, justification: string): string =>
    `Разрешить однократную запись файла вне рабочей области субагентом ${child} в ${target}? ${justification}`,
  pt: (child: string, target: string, justification: string): string =>
    `Permitir uma escrita fora da área de trabalho do subagente ${child} em ${target}? ${justification}`,
} as const

/**
 * Build the parent-session localized presentation for one forwarded ask.
 * @param childSessionId - the real delegated Session asking.
 * @param target - absolute target path named by the original tool call.
 * @param justification - the model's verbatim justification.
 * @returns one value per supported UI locale; `en` is mandatory.
 */
export function childWriteApprovalDisplayReason(
  childSessionId: string,
  target: string,
  justification: string,
): Record<keyof typeof LOCALES, string> {
  return {
    en: LOCALES.en(childSessionId, target, justification),
    zh: LOCALES.zh(childSessionId, target, justification),
    ja: LOCALES.ja(childSessionId, target, justification),
    es: LOCALES.es(childSessionId, target, justification),
    fr: LOCALES.fr(childSessionId, target, justification),
    de: LOCALES.de(childSessionId, target, justification),
    ru: LOCALES.ru(childSessionId, target, justification),
    pt: LOCALES.pt(childSessionId, target, justification),
  }
}

/** Read the child's effective workspace root for the containment test. */
function workspaceRootOf(child: Agent): string | undefined {
  const workingDirectory = child.ctx.get('workingDirectory')
  if (workingDirectory !== undefined) return workingDirectory.get(child.session)
  return child.session.header.cwd
}

/** Whether a resolved target stays at or below the workspace root. */
export function isInsideWorkspaceRoot(root: string, target: string): boolean {
  const caseSensitive = process.platform !== 'win32'
  const comparableRoot = caseSensitive ? root : root.toLowerCase()
  const comparableTarget = caseSensitive ? target : target.toLowerCase()
  const prefix = comparableRoot.endsWith(sep) ? comparableRoot : comparableRoot + sep
  if (comparableTarget === comparableRoot || comparableTarget.startsWith(prefix)) return true
  const fromRoot = relative(comparableRoot, comparableTarget)
  return fromRoot !== '..' && !fromRoot.startsWith('..' + sep) && !isAbsolute(fromRoot)
}

/** Parse one logged tool-call argument object without trusting its shape. */
function parseArguments(raw: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(raw)
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined
  } catch {
    return undefined
  }
}

/**
 * Decide whether one child approval request is the covered class: a
 * `danger-full-access` escalation of a single-file `write` whose logged call
 * arguments name one file outside the child's workspace. Anything else — another
 * tool, another target mode, an unrecorded or mismatched call id, a target inside
 * the workspace, a missing justification — is not covered.
 * @param child - the delegated Agent asking.
 * @param request - the pending approval request; its `callId` is the binding key.
 * @param callArgumentsJson - the raw argument string recorded for that call by the incremental ledger, if any.
 * @returns the resolved escalation, or `undefined` when the class does not match.
 */
export function readChildWriteEscalation(
  child: Agent,
  request: ApprovalRequestEvent,
  callArgumentsJson: string | undefined,
): ChildWriteEscalation | undefined {
  if (request.toolName !== 'write' || request.callId === undefined) return undefined
  if (request.agent !== child) return undefined
  const callId = request.callId
  if (callArgumentsJson === undefined) return undefined
  const args = parseArguments(callArgumentsJson)
  if (args === undefined) return undefined
  if (args['sandbox_permissions'] !== 'danger-full-access') return undefined
  const justification = args['justification']
  if (typeof justification !== 'string' || justification.trim() === '') return undefined
  const filePath = args['file_path']
  if (typeof filePath !== 'string' || filePath.trim() === '') return undefined
  // The write tool publishes one complete file; a call without content is not that tool.
  if (typeof args['content'] !== 'string') return undefined
  const root = workspaceRootOf(child)
  if (root === undefined) return undefined
  const target = resolve(root, filePath)
  // Only the genuine out-of-workspace case is covered: an escalation the sandbox
  // would not have required stays denied instead of consuming a user's attention.
  if (isInsideWorkspaceRoot(root, target)) return undefined
  return {
    callId,
    target,
    justification,
    argsDigest: createHash('sha256').update(callArgumentsJson).digest('hex'),
  }
}

/** The lineage the forwarded ask must still satisfy at decision time. */
function assertDelegationBinding(child: Agent, parent: Agent): boolean {
  if (child.session.header.parentSession !== parent.session.id) return false
  try {
    return delegationDepthOf(child) === delegationDepthOf(parent) + 1
  } catch {
    return false
  }
}

/** The parent's effective approval policy, read through the public service config and override. */
function parentPolicyOf(parent: Agent): 'ask' | 'never' | undefined {
  const approval = parent.ctx.get('approval')
  if (approval === undefined) return undefined
  return approval.overrideOf(parent.session) ?? approval.config.policy ?? 'ask'
}

/**
 * Report one refused or completed forwarding attempt through the process log the
 * deployment already reads; a refusal must be diagnosable without the model. The
 * durable record of the ask itself stays the child's own `approval/asked` +
 * `approval/decided` pair, so this line never fabricates an unpaired audit event.
 * @param child - the delegated Agent.
 * @param message - the concrete refusal or completion.
 */
function note(child: Agent, message: string): void {
  child.ctx.logger.warn(`subagent child-write-approval: ${message}`)
}

/** Cancellation lifetime of one forwarded ask: the request's own signal plus a fail-closed timeout. */
function askLifetime(request: ApprovalRequestEvent, timeoutMs: number): {
  readonly signal: AbortSignal
  readonly dispose: () => void
} {
  const controller = new AbortController()
  const timeout = setTimeout(() => {
    controller.abort(new Error('forwarded child write approval timed out'))
  }, timeoutMs)
  const signal = request.signal === undefined
    ? controller.signal
    : AbortSignal.any([request.signal, controller.signal])
  return {
    signal,
    dispose: () => { clearTimeout(timeout) },
  }
}

/**
 * Forward one covered escalation to the parent session's approval channel and
 * return the closed outcome the child's original call consumes. Every failure —
 * no composed approval service, a parent `never` policy, a closed parent turn,
 * a timeout, a withdrawn request, a changed deployment mode, or a call whose
 * logged arguments no longer match — resolves without a grant.
 * @param child - the delegated Agent asking.
 * @param parent - the live delegating Agent captured at delegation.
 * @param request - the child's pending approval request.
 * @param escalation - the covered escalation resolved from the child's tool call.
 * @param readSettings - the deployment settings, read now so an opt-out applies here.
 * @returns the child's closed outcome; only `allowed-once` is a grant.
 */
async function forwardToParent(
  child: Agent,
  parent: Agent,
  request: ApprovalRequestEvent,
  escalation: ChildWriteEscalation,
  readSettings: () => ChildWriteApprovalSettings,
  calls: ChildWriteCallLedger,
): Promise<ApprovalOutcome> {
  const settings = readSettings()
  if (settings.mode !== 'forward') return 'rejected'
  if (!assertDelegationBinding(child, parent)) return 'unavailable'
  if (parentPolicyOf(parent) !== 'ask') return 'rejected'
  const approval = parent.ctx.get('approval')
  if (approval === undefined) return 'unavailable'
  const lifetime = askLifetime(request, settings.timeoutMs)
  try {
    const outcome = await approval.request({
      agent: parent,
      toolName: request.toolName,
      reason: `subagent "${child.session.id}" requests one out-of-workspace file write to "${escalation.target}": `
        + escalation.justification,
      displayReason: childWriteApprovalDisplayReason(child.session.id, escalation.target, escalation.justification),
      signal: lifetime.signal,
    })
    if (outcome !== 'allowed-once') return outcome
    // Bind the grant to the exact original call and to the arguments the parent's
    // user was shown: a mismatch leaves the child denied.
    const recheck = readChildWriteEscalation(
      child,
      request,
      request.callId === undefined ? undefined : calls.get(request.callId),
    )
    if (recheck === undefined
      || recheck.callId !== escalation.callId
      || recheck.argsDigest !== escalation.argsDigest) {
      return 'unavailable'
    }
    return 'allowed-once'
  } catch {
    // A closed parent turn (the audit pair must be turn-enclosed), a disposed
    // parent, or an append failure all leave the child denied.
    return 'unavailable'
  } finally {
    lifetime.dispose()
  }
}

/**
 * Install the forwarder in one fresh delegated child's scope. It claims exactly
 * the covered class with a prepended waterfall listener (ahead of transport-wide
 * answerers, so the child's own session never asks independently); every other
 * ask from this child resolves `rejected`, preserving the pinned `never` stance.
 * @param child - the freshly created delegated Agent.
 * @param parent - its live delegating Agent.
 * @param readSettings - deployment settings read at ask time.
 * @param calls - the live tool-call ledger for this child, fed from delivered session events.
 */
export function installChildWriteApprovalForwarder(
  child: Agent,
  parent: Agent,
  readSettings: () => ChildWriteApprovalSettings,
  calls: ChildWriteCallLedger,
): void {
  child.ctx.on('approval/request', async (request, next) => {
    if (request.agent !== child) return next()
    const escalation = readChildWriteEscalation(
      child,
      request,
      request.callId === undefined ? undefined : calls.get(request.callId),
    )
    if (escalation === undefined) return 'rejected'
    const outcome = await forwardToParent(child, parent, request, escalation, readSettings, calls)
    note(child, outcome === 'allowed-once'
      ? `forwarded write approval granted once for "${escalation.target}"`
      : `forwarded write approval for "${escalation.target}" settled ${outcome}; the child stays denied`)
    return outcome
  }, { prepend: true })
}

/**
 * Install the fail-closed guard used for a cold-resumed delegated child: a
 * replayed delegation log may carry an `ask` policy, and a resumed child has no
 * captured forwarding route, so no approval request from it may reach any
 * answerer (the deployment's own transport-wide answerers included).
 * @param child - the resumed delegated Agent.
 */
export function installResumedChildApprovalGuard(child: Agent): void {
  child.ctx.on('approval/request', (request, next) => {
    if (request.agent !== child) return next()
    note(child, 'resumed delegated child has no captured approval route; every ask is denied')
    return Promise.resolve<ApprovalOutcome>('rejected')
  }, { prepend: true })
}
