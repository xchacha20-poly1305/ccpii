import type { Vault } from '../types'

/** A detected span, offsets in UTF-16 code units as JavaScript strings index. */
export type Span = { label: string; start: number; end: number }

/** openai/privacy-filter's labels and the placeholder kind each one becomes. */
export const KINDS: Readonly<Record<string, string>> = {
  account_number: 'ACCOUNT',
  private_address: 'ADDRESS',
  private_date: 'DATE',
  private_email: 'EMAIL',
  private_person: 'PERSON',
  private_phone: 'PHONE',
  private_url: 'URL',
  secret: 'SECRET',
}

const UNKNOWN_KIND = 'PII'

/**
 * The placeholder kind for a label: privacy-filter's own by KINDS, any other
 * backend's label as it is, upper-cased, each character outside A-Z, 0-9
 * and `_` made `_` (Presidio's `EMAIL_ADDRESS` stays `EMAIL_ADDRESS`).
 */
export function kindOf(label: string): string {
  const known = KINDS[label]
  if (known !== undefined) return known
  let kind = ''
  for (const char of label.toUpperCase()) {
    const code = char.charCodeAt(0)
    const isKept = (code >= 65 && code <= 90) || (code >= 48 && code <= 57) || code === 95
    kind += isKept && char.length === 1 ? char : '_'
  }

  return kind === '' ? UNKNOWN_KIND : kind
}

/**
 * What a placeholder is wrapped in: `⟦EMAIL_0⟧`. Text seldom holds these
 * brackets, so a bare `EMAIL_0` in a file stays what it is, and only the
 * placeholders this table issued are ever restored.
 */
export const OPEN = '⟦'
export const CLOSE = '⟧'

export const placeholderOf = (kind: string, number: number): string => `${OPEN}${kind}_${number}${CLOSE}`

export const emptyVault = (sessionId: string): Vault => ({
  sessionId,
  counters: {},
  byValue: {},
  byPlaceholder: {},
})

/**
 * Replaces each span of `text` with its placeholder, issuing new ones as
 * needed. Returns the vault unchanged (the same object) when nothing new was
 * issued, so a caller can tell whether it has anything to save.
 */
export function mask(
  vault: Vault,
  text: string,
  spans: readonly Span[],
): { vault: Vault; text: string } {
  const kept = withoutOverlaps(spans).filter(span => {
    const value = text.slice(span.start, span.end)

    return !holdsPlaceholder(vault, value) && !isHashTakenForAccount(span.label, value)
  })
  if (kept.length === 0) return { vault, text }

  let next = vault
  const issue = (kind: string, value: string): string => {
    const known = next.byValue[value]
    if (known !== undefined) return known
    if (next === vault) {
      next = {
        sessionId: vault.sessionId,
        counters: { ...vault.counters },
        byValue: { ...vault.byValue },
        byPlaceholder: { ...vault.byPlaceholder },
      }
    }
    const number = next.counters[kind] ?? 0
    const placeholder = placeholderOf(kind, number)
    next.counters[kind] = number + 1
    next.byValue[value] = placeholder
    next.byPlaceholder[placeholder] = value
    return placeholder
  }

  let masked = ''
  let cursor = 0
  for (const span of kept) {
    masked += text.slice(cursor, span.start)
    masked += issue(kindOf(span.label), text.slice(span.start, span.end))
    cursor = span.end
  }
  masked += text.slice(cursor)

  return { vault: next, text: masked }
}

/** Puts the real value back for every placeholder of the vault in `text`. */
export function unmask(vault: Vault, text: string): string {
  let restored = ''
  let cursor = 0
  for (const found of placeholders(vault, text)) {
    restored += text.slice(cursor, found.start) + found.value
    cursor = found.end
  }

  return cursor === 0 ? text : restored + text.slice(cursor)
}

/** `unmask` over every string inside a JSON-like value; unchanged parts are kept by identity. */
export function unmaskDeep<T>(vault: Vault, value: T): T {
  if (typeof value === 'string') return unmask(vault, value) as T
  if (Array.isArray(value)) {
    const items = value.map(item => unmaskDeep(vault, item))

    return (items.some((item, i) => item !== value[i]) ? items : value) as T
  }
  if (value !== null && typeof value === 'object') {
    let changed = false
    const entries = Object.entries(value).map(([key, item]) => {
      const restored = unmaskDeep(vault, item)
      changed ||= restored !== item

      return [key, restored] as const
    })

    return (changed ? Object.fromEntries(entries) : value) as T
  }

  return value
}

/** Spans in order, each one that overlaps an earlier one dropped. */
function withoutOverlaps(spans: readonly Span[]): Span[] {
  const sorted = [...spans]
    .filter(span => span.end > span.start)
    .sort((a, b) => a.start - b.start || b.end - a.end)
  const kept: Span[] = []
  for (const span of sorted) {
    const last = kept[kept.length - 1]
    if (last === undefined || span.start >= last.end) kept.push(span)
  }

  return kept
}

/**
 * Whether the model took a hex digest (a git commit, a sha256) for an account
 * number, as it does: account numbers are digits and letters beyond `f`.
 * A hex `secret` is still masked; a key is often written in hex.
 */
function isHashTakenForAccount(label: string, text: string): boolean {
  if (label !== 'account_number' || text.length < 7) return false
  let hasDigit = false
  let hasHexLetter = false
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i) | 0x20 // lower-case a letter, leave a digit
    if (code >= 48 && code <= 57) hasDigit = true
    else if (code >= 97 && code <= 102) hasHexLetter = true
    else return false
  }

  return hasDigit && hasHexLetter
}

/** Whether `text` already holds one of the vault's placeholders: masking it again would hide the placeholder itself. */
function holdsPlaceholder(vault: Vault, text: string): boolean {
  return !placeholders(vault, text).next().done
}

/** Each of the vault's placeholders in `text`, in order, with the value it stands for. */
function* placeholders(vault: Vault, text: string): Generator<{ start: number; end: number; value: string }> {
  for (let start = text.indexOf(OPEN); start >= 0; start = text.indexOf(OPEN, start + 1)) {
    const close = text.indexOf(CLOSE, start + 1)
    if (close < 0) return
    const value = vault.byPlaceholder[text.slice(start, close + 1)]
    if (value === undefined) continue
    yield { start, end: close + 1, value }
    start = close
  }
}
