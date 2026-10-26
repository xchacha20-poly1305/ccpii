import { kindOf } from './vault'
import type { Span } from './vault'

/**
 * Which detected spans are masked: all of them but what the person left
 * unmasked in the options and what the model is known to mistake.
 */
export type Policy = {
  /** Placeholder kinds never masked (`PERSON`, `URL`). */
  unmaskedKinds: ReadonlySet<string>
  /** Words never masked, lower-cased. */
  unmaskedWords: readonly string[]
}

/**
 * The policy of the two options, each a comma-separated list. A kind is
 * written as its placeholder kind (`PERSON`) or as the detector's label
 * (`private_person`), in any case.
 */
export function policyOf(kinds: string, words: string): Policy {
  return {
    unmaskedKinds: new Set(listOf(kinds).map(kindOf)),
    unmaskedWords: listOf(words).map(word => word.toLowerCase()),
  }
}

/** What the policy leaves unmasked, for `/ccpii status`; undefined when it is nothing but the model's mistakes. */
export function describePolicy(policy: Policy): string | undefined {
  const parts = [
    ...(policy.unmaskedKinds.size === 0 ? [] : [`kinds ${[...policy.unmaskedKinds].join(', ')}`]),
    ...(policy.unmaskedWords.length === 0 ? [] : [`words ${policy.unmaskedWords.join(', ')}`]),
  ]

  return parts.length === 0 ? undefined : parts.join('; ')
}

/** The spans of `text` the policy masks. */
export function maskedSpans(policy: Policy, text: string, spans: readonly Span[]): Span[] {
  return spans.filter(span => {
    const value = text.slice(span.start, span.end)

    return (
      !policy.unmaskedKinds.has(kindOf(span.label)) &&
      !holdsUnmaskedWord(policy, value) &&
      !isHashTakenForAccount(span.label, value) &&
      !isPathTakenForUrl(span.label, value) &&
      !isModeTakenForPerson(span.label, value)
    )
  })
}

function listOf(option: string): string[] {
  return option
    .split(',')
    .map(item => item.trim())
    .filter(item => item !== '')
}

/**
 * Whether `text` holds one of the unmasked words, in any case. Where an end
 * of the word and the character beside it are both ASCII letters or digits,
 * the word runs on and does not count (`ann` is not in `Joanna`); Chinese
 * has no spaces between words, so `张伟` counts in `张伟明天`.
 */
function holdsUnmaskedWord(policy: Policy, text: string): boolean {
  const lower = text.toLowerCase()

  return policy.unmaskedWords.some(word => {
    for (let at = lower.indexOf(word); at >= 0; at = lower.indexOf(word, at + 1)) {
      const before = lower[at - 1]
      const after = lower[at + word.length]
      const runsOn = (beside: string | undefined, edge: string | undefined) =>
        beside !== undefined && edge !== undefined && isAsciiWordChar(beside) && isAsciiWordChar(edge)
      if (!runsOn(before, word[0]) && !runsOn(after, word[word.length - 1])) return true
    }

    return false
  })
}

function isAsciiWordChar(char: string): boolean {
  const code = char.charCodeAt(0) | 0x20 // lower-case a letter, leave a digit
  return (code >= 48 && code <= 57) || (code >= 97 && code <= 122)
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

/**
 * Whether the model took a file path for a URL, as it does when a text is a
 * path and nothing else (`pwd`, a listing of one file). A path starts at the
 * root, the home or the current directory, or a drive letter; or else its
 * first segment, unlike a host, has no dot, port or user (`hooks/vault.ts`
 * against `example.com/a`). A URL with a scheme is never a path.
 */
function isPathTakenForUrl(label: string, text: string): boolean {
  if (label !== 'private_url' || text.includes('://') || text.startsWith('//')) return false
  if (text.startsWith('/') || text.startsWith('~') || text.startsWith('.')) return true
  if (text[1] === ':' && (text[2] === '\\' || text[2] === '/')) return true
  const slash = text.indexOf('/')
  if (slash < 0) return false
  const first = text.slice(0, slash)

  return !first.includes('.') && !first.includes(':') && !first.includes('@')
}

/** Whether the model took a file mode in `ls -l` output (`drwxr-xr-x`) for a person, as it does. */
function isModeTakenForPerson(label: string, text: string): boolean {
  if (label !== 'private_person' || text.length !== 10 || !'-dlcbps'.includes(text[0] ?? '')) return false
  for (let i = 1; i < text.length; i++) {
    if (!'rwxsStT-'.includes(text[i] ?? '')) return false
  }

  return true
}
