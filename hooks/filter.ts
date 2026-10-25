import type { EngineInterface } from 'claude-code'

import type { Span } from './vault'

export type FilterStatus =
  /** Not started by this module yet. */
  | 'idle'
  /** Spawned; the socket is not bound yet. */
  | 'starting'
  /** The socket is bound and the model loads; requests wait for it. */
  | 'loading'
  | 'ready'
  /** The checkpoint is not installed; nothing is retried until `/ccpii install`. */
  | 'missing'
  | 'installing'
  /** The model failed to load; the process lives on, answering errors. */
  | 'failed'
  /** The process is gone. */
  | 'exited'
  /** No sidecar: the `endpoint` option names a detection service of the person's own. */
  | 'external'

const sidecarDirRef = { plugin: 'ccpii', key: 'sidecarDir' } as const

/** How long a request waits for a starting sidecar to bind its socket. */
const LISTEN_WAIT_MS = 5_000

/**
 * The privacy-filter sidecar this module runs: a Python process serving
 * `POST /detect` on a Unix socket in a private directory, or the install
 * that precedes it. One child at a time; each start or install supersedes
 * the one before, and a superseded child's output changes nothing.
 */
const sidecar = {
  status: 'idle' as FilterStatus,
  /** What went wrong, for `missing`, `failed` and `exited`. */
  error: undefined as string | undefined,
  onChange: () => {},
  /** The socket the current child named on its `listening` line. */
  socketPath: undefined as string | undefined,
  listening: Promise.resolve(),
  stopChild: undefined as (() => void) | undefined,
  generation: 0,
  /**
   * Whether the current child got as far as `ready`. One that did and then
   * died is started again by the next request; one that died before it was
   * ready would most likely die again, so it waits for /ccpii restart.
   */
  wasReady: false,
  endpoint: undefined as string | undefined,
}

/**
 * `endpoint`, when not empty, is the URL of a detection service that answers
 * `POST {"texts"}` as the sidecar does (BACKENDS.md); no sidecar is run then.
 */
export function configureFilter(onChange: () => void, endpoint: string): void {
  sidecar.onChange = onChange
  sidecar.endpoint = endpoint.trim() === '' ? undefined : endpoint.trim()
}

export function filterStatus(): FilterStatus {
  return sidecar.status
}

export async function startFilter($: EngineInterface): Promise<void> {
  const generation = supersede()
  if (sidecar.endpoint !== undefined) {
    report(generation, 'external')
    return
  }
  sidecar.wasReady = false
  // A child stopped by a signal it cannot catch (a crash, or this module
  // unloading) leaves its directory behind; `$.state` outlives a reload.
  sidecar.socketPath = undefined
  const { value: previousDir } = await $.state.get(sidecarDirRef)
  if (previousDir !== undefined && isSidecarDir(previousDir)) await $.process.run(['rm', '-rf', previousDir])
  let markListening = () => {}
  sidecar.listening = new Promise(resolve => (markListening = resolve))
  report(generation, 'starting')

  spawnScript($, generation, ['--serve'], {
    onLine: (word, rest) => {
      // Bound means requests are taken: each waits for the model to load.
      if (word === 'listening') {
        sidecar.socketPath = rest
        // Lost, it costs one empty directory left after the next reload.
        $.state.set(sidecarDirRef, rest.slice(0, rest.lastIndexOf('/'))).catch(() => {})
        report(generation, 'loading')
        markListening()
      }
      if (word === 'ready') sidecar.wasReady = true
      if (word === 'ready' || word === 'missing') report(generation, word)
      if (word === 'failed') report(generation, 'failed', rest)
    },
    onExit: reason => {
      markListening()
      if (sidecar.status !== 'missing') report(generation, 'exited', reason)
    },
  })
}

export async function installFilter($: EngineInterface): Promise<void> {
  if (sidecar.status === 'installing' || sidecar.status === 'external') return
  const generation = supersede()
  report(generation, 'installing')

  spawnScript($, generation, ['--download'], {
    onLine: (word, rest) => {
      if (word === 'installed') startFilter($).catch(error => report(generation, 'exited', String(error)))
      if (word === 'failed') report(generation, 'missing', `the install failed: ${rest}`)
    },
    onExit: reason => {
      if (sidecar.status === 'installing') report(generation, 'missing', `the install stopped: ${reason}`)
    },
  })
}

/** The spans of each text, in order; throws, saying why in plain words, when the sidecar cannot answer. */
/**
 * The spans in each of `texts`, offsets inside its text.
 *
 * The host ends a fetch after 30 s, with no way to ask for longer, so no
 * request carries more than `REQUEST_UNITS`: a longer text goes in pieces,
 * the pieces in as many requests as they need, and the requests one at a
 * time across all callers, since the filter answers one after another and a
 * request waiting behind another's would spend its own 30 s waiting.
 */
export async function detect($: EngineInterface, texts: readonly string[]): Promise<Span[][]> {
  const spans: Span[][] = texts.map(() => [])
  for (const request of packRequests(texts.flatMap(cutPieces))) {
    const answers = await inTurn(() => post($, request.map(piece => piece.body)))
    request.forEach((piece, i) => {
      for (const span of answers[i] ?? []) {
        const start = piece.start + span.start
        // A span belongs to the piece whose own part it starts in; the
        // overlap of the next or previous piece sees it too.
        if (start < piece.ownStart || start >= piece.ownEnd) continue
        spans[piece.text]?.push({ label: span.label, start, end: piece.start + span.end })
      }
    })
  }

  return spans
}

/**
 * The most one request carries, in UTF-16 code units: the bundled filter
 * answers that in about 4 s on a laptop CPU, well inside the 30 s.
 */
const REQUEST_UNITS = 16_384

/**
 * Text a piece carries past its own part on either side, so a span near a
 * cut is seen whole and with its context. The model looks 128 tokens either
 * way; 1,024 units hold that in any script.
 */
const OVERLAP_UNITS = 1_024

/**
 * Part of one text: `body` is `text.slice(start, ...)`, and the spans kept
 * from it are those starting in `[ownStart, ownEnd)`.
 */
type Piece = { text: number; start: number; ownStart: number; ownEnd: number; body: string }

function cutPieces(text: string, index: number): Piece[] {
  if (text.length <= REQUEST_UNITS) {
    return text === '' ? [] : [{ text: index, start: 0, ownStart: 0, ownEnd: text.length, body: text }]
  }
  const pieces: Piece[] = []
  for (let ownStart = 0; ownStart < text.length; ) {
    const ownEnd = cutAfter(text, ownStart, REQUEST_UNITS - 2 * OVERLAP_UNITS)
    const start = boundary(text, Math.max(0, ownStart - OVERLAP_UNITS))
    const end = boundary(text, Math.min(text.length, ownEnd + OVERLAP_UNITS))
    pieces.push({ text: index, start, ownStart, ownEnd, body: text.slice(start, end) })
    ownStart = ownEnd
  }

  return pieces
}

/**
 * Where a piece's own part starting at `from` ends: after the last line
 * break within `limit` units when that keeps at least half of them, else at
 * `limit`, never between the halves of a surrogate pair.
 */
function cutAfter(text: string, from: number, limit: number): number {
  if (text.length - from <= limit) return text.length
  const lineEnd = text.lastIndexOf('\n', from + limit - 1)

  return lineEnd >= from + limit / 2 ? lineEnd + 1 : boundary(text, from + limit)
}

/** `at`, or one before it when `at` would split a surrogate pair. */
function boundary(text: string, at: number): number {
  const code = text.charCodeAt(at - 1)

  return at > 0 && at < text.length && code >= 0xd800 && code <= 0xdbff ? at - 1 : at
}

/** Pieces in order, grouped into requests of at most `REQUEST_UNITS`. */
function packRequests(pieces: Piece[]): Piece[][] {
  const requests: Piece[][] = []
  let units = REQUEST_UNITS
  for (const piece of pieces) {
    if (units + piece.body.length > REQUEST_UNITS) {
      requests.push([])
      units = 0
    }
    requests[requests.length - 1]?.push(piece)
    units += piece.body.length
  }

  return requests
}

/** The last request handed to `inTurn`; the next one starts after it settles. */
let queue: Promise<unknown> = Promise.resolve()

function inTurn<T>(request: () => Promise<T>): Promise<T> {
  const turn = queue.then(request)
  queue = turn.catch(() => {})

  return turn
}

/** One `POST {"texts"}` to the endpoint or the sidecar, its answer checked. */
async function post($: EngineInterface, texts: readonly string[]): Promise<Span[][]> {
  if (sidecar.status === 'external' && sidecar.endpoint !== undefined) {
    const response = await $.http.fetch(sidecar.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts }),
    })
    if (!response.ok) throw new Error(`the detection service answered ${response.status}: ${response.text.slice(0, 200)}`)

    return readSpans(response.text, texts, 'the detection service')
  }
  if (sidecar.status === 'exited' && sidecar.wasReady) {
    await startFilter($)
  }
  if (sidecar.status === 'starting') await Promise.race([sidecar.listening, $.clock.sleep(LISTEN_WAIT_MS)])
  if ((sidecar.status !== 'loading' && sidecar.status !== 'ready') || sidecar.socketPath === undefined) {
    throw new Error(describeFilter())
  }

  const response = await $.http.fetch('http://ccpii/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts }),
    socketPath: sidecar.socketPath,
  })
  if (!response.ok) throw new Error(`the privacy filter answered ${response.status}: ${response.text}`)

  return readSpans(response.text, texts, 'the privacy filter')
}

/**
 * The spans of a `/detect` answer, checked against the texts they are for:
 * a list per text, each span a label and offsets inside its text. Any other
 * answer throws, so what it was about is withheld rather than sent unchecked.
 */
function readSpans(body: string, texts: readonly string[], who: string): Span[][] {
  let parsed: unknown
  try {
    parsed = JSON.parse(body)
  } catch {
    throw new Error(`${who} answered something other than JSON`)
  }
  const spans = (parsed as { spans?: unknown } | null)?.spans
  if (!Array.isArray(spans) || spans.length !== texts.length) {
    throw new Error(`${who} answered no "spans" list with one entry per text`)
  }

  return spans.map((list: unknown, i) => {
    const length = texts[i]?.length ?? 0
    if (!Array.isArray(list)) throw new Error(`${who} answered spans that are not a list`)

    return list.map((span: unknown) => {
      const { label, start, end } = (span ?? {}) as Partial<Span>
      const isValid =
        typeof label === 'string' &&
        Number.isInteger(start) &&
        Number.isInteger(end) &&
        (start as number) >= 0 &&
        (start as number) < (end as number) &&
        (end as number) <= length
      if (!isValid) throw new Error(`${who} answered a span that is not { label, start, end } inside its text`)

      return { label, start, end } as Span
    })
  })
}

/** The status in a sentence the person can act on. */
export function describeFilter(): string {
  switch (sidecar.status) {
    case 'idle':
      return 'the privacy filter has not started'
    case 'starting':
      return 'the privacy filter is still starting'
    case 'loading':
      return 'the privacy-filter model is loading'
    case 'ready':
      return 'the privacy filter is ready'
    case 'missing':
      return `the privacy-filter model is not installed${sidecar.error ? ` (${sidecar.error})` : ''}; run /ccpii install`
    case 'installing':
      return 'the privacy-filter model is being installed; see /ccpii status'
    case 'failed':
      return `the privacy-filter model failed to load (${sidecar.error}); run /ccpii restart`
    case 'external':
      return `the detection service at ${sidecar.endpoint} is in use`
    case 'exited':
      return sidecar.wasReady
        ? `the privacy filter stopped (${sidecar.error}); the next request starts it again`
        : `the privacy filter stopped before it was ready (${sidecar.error}); run /ccpii restart`
  }
}

/** Ends the current child and returns the generation of the next. */
function supersede(): number {
  sidecar.stopChild?.()
  sidecar.stopChild = undefined

  return ++sidecar.generation
}

function report(generation: number, status: FilterStatus, error?: string): void {
  if (generation !== sidecar.generation) return
  sidecar.status = status
  sidecar.error = error
  sidecar.onChange()
}

/**
 * Runs the sidecar script with `args` for as long as the module lives,
 * handing each stdout line on as its first word and the rest.
 */
function spawnScript(
  $: EngineInterface,
  generation: number,
  args: readonly string[],
  handlers: { onLine: (word: string, rest: string) => void; onExit: (reason: string) => void },
): void {
  const child = $.process.spawn({
    argv: ['uv', 'run', '--quiet', '--script', `${$.plugin.root}/server/pii_server.py`, ...args],
  })
  sidecar.stopChild = () => void child.return({ code: null, signal: null })
  const isCurrent = () => generation === sidecar.generation

  void (async () => {
    let pending = ''
    let stderrTail = ''
    try {
      for await (const { stream, text } of child) {
        if (stream === 'stderr') {
          $.ui.log(text.trimEnd(), { to: 'debug' })
          stderrTail = (stderrTail + text).slice(-300)
          continue
        }
        pending += text
        for (let newline = pending.indexOf('\n'); newline >= 0; newline = pending.indexOf('\n')) {
          const line = pending.slice(0, newline).trim()
          pending = pending.slice(newline + 1)
          const space = line.indexOf(' ')
          if (isCurrent()) handlers.onLine(space < 0 ? line : line.slice(0, space), space < 0 ? '' : line.slice(space + 1))
        }
      }
      const { code, signal } = await child.result
      const how = exitReason(code, signal)
      const said = lastLine(stderrTail)
      if (isCurrent()) handlers.onExit(said === '' ? how : `${how}: ${said}`)
    } catch (error) {
      if (isCurrent()) handlers.onExit(error instanceof Error ? error.message : String(error))
    }
  })()
}

/**
 * How a child ended, in words. The child is `uv`, which exits with 128 + N
 * when the Python process under it dies of signal N.
 */
function exitReason(code: number | null, signal: string | null): string {
  if (signal !== null) return `it was killed by ${signal}`
  if (code !== null && code > 128) return `it was killed by ${SIGNALS[code - 128] ?? `signal ${code - 128}`}`

  return `it exited with code ${code}`
}

const SIGNALS: Readonly<Record<number, string>> = {
  1: 'SIGHUP',
  2: 'SIGINT',
  6: 'SIGABRT',
  9: 'SIGKILL',
  11: 'SIGSEGV',
  15: 'SIGTERM',
}

/** A directory the sidecar makes: `/tmp/ccpii.<random>`, nothing above or beside it. */
function isSidecarDir(path: string): boolean {
  const name = path.slice('/tmp/'.length)

  return path.startsWith('/tmp/ccpii.') && name.length > 'ccpii.'.length && !name.includes('/') && !name.includes('..')
}

function lastLine(text: string): string {
  const lines = text.trim().split('\n')

  return (lines[lines.length - 1] ?? '').trim()
}
