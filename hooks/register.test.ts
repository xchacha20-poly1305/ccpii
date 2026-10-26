import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { placeholderOf as ph } from './vault'

type Detected = { label: string; start: number; end: number }

/** What the sidecar script does when the mod runs it. */
type Script = {
  /**
   * Lines the serving process prints, by how many starts there have been;
   * it then runs until the test ends, unless a line is `missing` or `crash`.
   */
  serve: string[] | ((started: number) => string[])
  /** Lines `--download` prints before it exits. */
  download?: string[]
  /** Whether `/detect` answers. */
  isAnswering?: boolean
  /** The label the detector gives each address; privacy-filter's by default. */
  label?: string
  /** Answers spans past the end of their text, as a broken backend might. */
  isAnswerBroken?: boolean
}

/**
 * Stands for the engine and the sidecar: a filter that finds each address in
 * `emails` wherever it occurs, and a tool that records the command it ran.
 */
function engine(on: On, emails: string[], script: Script) {
  const ran: string[] = []
  const spawned: string[][] = []
  const commands: string[][] = []
  const fetched: { url: string; socketPath?: string }[] = []
  /** The texts of each request, and the most requests the filter held at once. */
  const requests: string[][] = []
  const load = { inFlight: 0, most: 0 }
  let stop = () => {}
  const stopped = new Promise<void>(resolve => (stop = resolve))
  const clock = mock.clock(on)
  mock.store(on)
  const session = { id: 'session-1' }

  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('command.register', (_, e) => ({ value: { command: e.name } }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('session.id', () => ({ value: session.id }))
  on('session.messages', () => ({ value: [] }))
  on('env.get', () => ({ value: '/home/someone' }))
  on('fs.exists', () => ({ value: false }))
  on('fs.write', () => ({ value: undefined }))
  on('process.run', (_, e) => {
    commands.push([...e.argv])

    return { value: { exitCode: 0, stdout: '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('process.spawn', async function* (_, e) {
    // argv: uv run --quiet --script <path> <args...>
    const args = e.argv.slice(5)
    spawned.push([...args])
    if (args.includes('--download')) {
      for (const line of script.download ?? []) yield { stream: 'stdout' as const, text: `${line}\n` }

      return { value: { code: 0, signal: null } }
    }
    const lines = typeof script.serve === 'function' ? script.serve(spawned.length) : script.serve
    for (const line of lines) yield { stream: 'stdout' as const, text: `${line}\n` }
    if (lines.includes('crash')) return { value: { code: null, signal: 'SIGKILL' } }
    if (!lines.includes('missing')) await stopped

    return { value: { code: 0, signal: null } }
  })
  on('http.fetch', async (_, e) => {
    fetched.push({ url: e.url, socketPath: e.init?.socketPath })
    if (script.isAnswering === false) {
      return { value: { status: 500, ok: false, headers: {}, text: '{"error":"model not loaded"}' } }
    }
    const { texts } = JSON.parse(e.init?.body ?? '{}') as { texts: string[] }
    requests.push(texts)
    load.most = Math.max(load.most, ++load.inFlight)
    await Promise.resolve()
    load.inFlight -= 1
    const spans = texts.map(text => {
      const found: Detected[] = []
      for (const email of emails) {
        for (let at = text.indexOf(email); at >= 0; at = text.indexOf(email, at + 1)) {
          found.push({ label: script.label ?? 'private_email', start: at, end: at + email.length })
        }
      }

      return script.isAnswerBroken ? found.map(span => ({ ...span, end: text.length + 5 })) : found
    })

    return { value: { status: 200, ok: true, headers: {}, text: JSON.stringify({ spans }) } }
  })
  on('tool.call', (_, e) => {
    if (e.tool === 'Bash') ran.push(e.command)

    return { result: { stdout: '', stderr: '', interrupted: false } }
  })

  return { ran, spawned, commands, fetched, requests, load, clock, session, stop }
}

const UP = ['listening /tmp/ccpii.test/filter.sock', 'loading', 'ready']

const ROW = {
  message: { type: 'user' as const, content: [{ type: 'text', text: 'write to alice@x.io' }] },
  door: 'tool-result' as const,
  origin: { kind: 'tool' as const, tool: 'Bash' },
  uuid: 'row-1',
}

const ccpii = (args: string) => ({
  command: 'ccpii',
  args,
  origin: { kind: 'composer' as const },
  presentation: { isFullscreen: false, columns: 80 },
})

const start = { cwd: '/', surface: 'terminal' as const, isInteractive: true }

test('a row the model reads is masked, and a tool call gets the real value back', async ($, on) => {
  const { ran, stop } = engine(on, ['alice@x.io'], { serve: UP })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)
  await $.tool.call({ tool: 'Bash', command: `echo ${ph('EMAIL', 0)}` })

  expect(JSON.stringify(rows.appended())).toContain(`write to ${ph('EMAIL', 0)}`)
  expect(JSON.stringify(rows.appended())).not.toContain('alice@x.io')
  expect(ran).toEqual(['echo alice@x.io'])
  stop()
})

test('the system prompt explains the placeholders', async ($, on) => {
  const { stop } = engine(on, [], { serve: UP })
  on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'You are Claude Code.', scope: 'shared' as const }] }))

  const { sections } = await $.prompt.compose({
    model: 'claude-opus-5-5',
    promptModel: 'claude-opus-5-5',
    surfaces: [],
    tools: [],
    outputStyle: null,
    traits: [],
  })

  expect(sections.map(section => section.id)).toEqual(['intro', 'ccpii:notice'])
  expect(sections[1]?.text).toContain(ph('EMAIL', 0))
  stop()
})

test('with the model not installed, a row is withheld and says how to install it', async ($, on) => {
  const { spawned, stop } = engine(on, ['alice@x.io'], { serve: ['missing'] })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)

  const stored = JSON.stringify(rows.appended())
  expect(stored).toContain('[ccpii] Withheld: the privacy-filter model is not installed')
  expect(stored).toContain('/ccpii install')
  expect(stored).not.toContain('alice@x.io')
  // A missing model is not retried: only the one start.
  expect(spawned).toEqual([['--serve']])
  stop()
})

test('a prompt the filter cannot check is dropped, with the reason', async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: UP, isAnswering: false })
  await $.session.start(start)

  const submitted = await $.prompt.submit({ text: 'mail alice@x.io', wait: false, origin: { kind: 'composer' } })

  expect(submitted.drop).toContain('answered 500')
  stop()
})

test('/ccpii install downloads, then serves', async ($, on) => {
  let isInstalled = false
  const { spawned, clock, stop } = engine(on, ['alice@x.io'], {
    serve: () => (isInstalled ? UP : ['missing']),
    download: ['downloading', 'installed'],
  })
  const rows = mock.session(on)
  await $.session.start(start)

  isInstalled = true
  await $.command.run(ccpii('install'))
  await clock.advance(0)
  await $.session.append(ROW)

  expect(spawned.map(args => args[0])).toEqual(['--serve', '--download', '--serve'])
  expect(JSON.stringify(rows.appended())).toContain(`write to ${ph('EMAIL', 0)}`)
  stop()
})

test('/ccpii pause lets content through until /ccpii resume', async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: ['missing'] })
  const rows = mock.session(on)
  await $.session.start(start)

  const paused = await $.command.run(ccpii('pause'))
  await $.session.append(ROW)
  await $.command.run(ccpii('resume'))
  await $.session.append({ ...ROW, uuid: 'row-2' })

  expect(paused.text).toContain('unfiltered')
  const [first, second] = rows.appended().map(row => JSON.stringify(row.message))
  expect(first).toContain('alice@x.io')
  expect(second).toContain('Withheld')
  stop()
})

test('a filter that was ready and then died is started again by the next request', async ($, on) => {
  const { spawned, commands, clock, stop } = engine(on, ['alice@x.io'], {
    serve: started => (started === 1 ? ['listening /tmp/ccpii.dead/filter.sock', 'ready', 'crash'] : UP),
  })
  const rows = mock.session(on)
  await $.session.start(start)
  await clock.advance(0)

  await $.session.append(ROW)

  expect(JSON.stringify(rows.appended())).toContain(`write to ${ph('EMAIL', 0)}`)
  expect(spawned.length).toBe(2)
  // The dead child's directory is removed before the next one starts.
  expect(commands).toContainEqual(['rm', '-rf', '/tmp/ccpii.dead'])
  stop()
})

test('a filter that died before it was ready is not retried, and says so', async ($, on) => {
  const { spawned, clock, stop } = engine(on, ['alice@x.io'], {
    serve: ['listening /tmp/ccpii.dead/filter.sock', 'crash'],
  })
  const rows = mock.session(on)
  await $.session.start(start)
  await clock.advance(0)

  await $.session.append(ROW)
  await $.session.append({ ...ROW, uuid: 'row-2' })

  const stored = JSON.stringify(rows.appended())
  expect(stored).toContain('stopped before it was ready (it was killed by SIGKILL); run /ccpii restart')
  expect(stored).not.toContain('alice@x.io')
  expect(spawned.length).toBe(1)
  stop()
})

test('/ccpii output carries no prefix of its own: the engine names the plugin', async ($, on) => {
  const { stop } = engine(on, [], { serve: UP })
  await $.session.start(start)

  const { text } = await $.command.run(ccpii('pause'))

  expect(text?.startsWith('Filtering paused')).toBe(true)
  stop()
})

const ENDPOINT = { options: { endpoint: 'http://127.0.0.1:8765/detect' } }

test('with an endpoint set, no sidecar runs and its labels become kinds', ENDPOINT, async ($, on) => {
  const { spawned, fetched, stop } = engine(on, ['alice@x.io'], { serve: UP, label: 'EMAIL_ADDRESS' })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)
  const { text } = await $.command.run(ccpii('install'))

  expect(spawned).toEqual([])
  expect(fetched).toEqual([{ url: 'http://127.0.0.1:8765/detect', socketPath: undefined }])
  expect(JSON.stringify(rows.appended())).toContain(`write to ${ph('EMAIL_ADDRESS', 0)}`)
  expect(text).toContain('Nothing to install')
  stop()
})

test('an endpoint whose answer does not fit its texts has the row withheld', ENDPOINT, async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: UP, isAnswerBroken: true })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)

  const stored = JSON.stringify(rows.appended())
  expect(stored).toContain('the detection service answered a span that is not { label, start, end } inside its text')
  expect(stored).not.toContain('alice@x.io')
  stop()
})

const isNotice = (row: { message: { content: readonly Record<string, unknown>[] } }) =>
  row.message.content.some(block => typeof block.text === 'string' && block.text.startsWith('# Privacy placeholders'))

const COMPOSE = {
  model: 'claude-opus-5-5',
  promptModel: 'claude-opus-5-5',
  surfaces: [],
  tools: [],
  outputStyle: null,
  traits: [],
} as const

test('with no notice in the system prompt, the first placeholders bring it in, once', async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: UP })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)
  await $.session.append({ ...ROW, uuid: 'row-2' })

  expect(rows.appended().map(isNotice)).toEqual([false, true, false])
  stop()
})

test('a conversation whose system prompt has the notice gets no note', async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: UP })
  on('prompt.compose', () => ({ sections: [{ id: 'intro', text: 'You are Claude Code.', scope: 'shared' as const }] }))
  const rows = mock.session(on)
  await $.session.start(start)

  await $.prompt.compose(COMPOSE)
  await $.session.append(ROW)

  expect(rows.appended().map(isNotice)).toEqual([false])
  stop()
})

test('a masked prompt carries the notice as context, the first time only', async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: UP })
  on('prompt.submit', (_, e) => ({ text: e.text, context: e.context }))
  await $.session.start(start)

  const first = await $.prompt.submit({ text: 'mail alice@x.io', wait: false, origin: { kind: 'composer' } })
  const second = await $.prompt.submit({ text: 'and alice@x.io again', wait: false, origin: { kind: 'composer' } })

  expect(first.text).toBe(`mail ${ph('EMAIL', 0)}`)
  expect(first.context?.[0]?.startsWith('# Privacy placeholders')).toBe(true)
  expect(second.context).toBeUndefined()
  stop()
})

test('after /clear, the new conversation gets its own notice', async ($, on) => {
  const { session, stop } = engine(on, ['alice@x.io'], { serve: UP })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)
  session.id = 'session-2'
  await $.session.append({ ...ROW, uuid: 'row-2' })

  expect(rows.appended().map(isNotice)).toEqual([false, true, false, true])
  stop()
})

/** A tool result row holding `text`. */
const rowOf = (text: string, uuid = 'row-long') => ({ ...ROW, uuid, message: { ...ROW.message, content: [{ type: 'text', text }] } })

const textOf = (row: { message: { content: readonly Record<string, unknown>[] } }) =>
  row.message.content.map(block => block.text).join('')

test('a long text goes in pieces of bounded requests, and a span across a cut is found once', async ($, on) => {
  const { requests, stop } = engine(on, ['alice@x.io', 'bob@x.io'], { serve: UP })
  const rows = mock.session(on)
  await $.session.start(start)
  // No line breaks, so the pieces are cut at their size, 14,336 units: the
  // first address straddles the first cut, the second sits in the last piece.
  const text = 'x'.repeat(14_330) + 'alice@x.io' + 'x'.repeat(30_000) + 'bob@x.io' + 'y'.repeat(100) + 'alice@x.io'

  await $.session.append(rowOf(text))

  const masked = textOf(rows.appended()[0]!)
  expect(masked).toBe(
    'x'.repeat(14_330) + ph('EMAIL', 0) + 'x'.repeat(30_000) + ph('EMAIL', 1) + 'y'.repeat(100) + ph('EMAIL', 0),
  )
  expect(requests.length).toBeGreaterThan(1)
  for (const texts of requests) expect(texts.join('').length).toBeLessThanOrEqual(16_384)
  stop()
})

test('a long text is cut after a line break near the size', async ($, on) => {
  const { requests, stop } = engine(on, ['alice@x.io'], { serve: UP })
  const rows = mock.session(on)
  await $.session.start(start)
  const line = 'z'.repeat(99) + '\n'
  const text = line.repeat(400) + 'alice@x.io\n'

  await $.session.append(rowOf(text))

  expect(textOf(rows.appended()[0]!)).toBe(line.repeat(400) + ph('EMAIL', 0) + '\n')
  // The first piece's own part ends on a line break, so the second piece,
  // which starts 1,024 units before its own part, has the break just before.
  const second = requests.flat()[1]!
  expect(second[1_023]).toBe('\n')
  stop()
})

test('short texts share one request, and requests from rows appended together run one at a time', async ($, on) => {
  const { requests, load, stop } = engine(on, ['alice@x.io'], { serve: UP })
  mock.session(on)
  await $.session.start(start)
  const long = 'q'.repeat(20_000)

  await Promise.all([$.session.append(rowOf(long, 'a')), $.session.append(rowOf(long, 'b')), $.session.append(ROW)])

  expect(load.most).toBe(1)
  expect(requests.flat().join('')).toContain('write to alice@x.io')
  stop()
})

test('a filter on the GPU says so, and a GPU fallback is counted with its reason', async ($, on) => {
  const { stop, clock } = engine(on, [], {
    serve: started =>
      started === 1
        ? [...UP, 'gpu starting', 'device gpu']
        : [...UP, 'gpu starting', 'device gpu', 'fallback the GPU gave no answer within 5 s', 'device cpu'],
  })
  await $.session.start(start)
  await clock.advance(0)
  const onGpu = (await $.command.run(ccpii('status'))).text
  await $.command.run(ccpii('restart'))
  await clock.advance(0)
  const afterFallback = (await $.command.run(ccpii('status'))).text

  expect(onGpu).toContain('filter: the privacy filter is ready (GPU)')
  expect(onGpu).toContain('device: the GPU answers')
  expect(afterFallback).toContain('filter: the privacy filter is ready (CPU)')
  expect(afterFallback).toContain(
    'device: the CPU answers until the GPU worker is ready; fell back to the CPU once, last: the GPU gave no answer within 5 s',
  )
  stop()
})

test('without a GPU model, status points to GPU.md', async ($, on) => {
  const { stop, clock } = engine(on, [], { serve: UP })
  await $.session.start(start)
  await clock.advance(0)

  expect((await $.command.run(ccpii('status'))).text).toContain(
    'device: the CPU answers; GPU.md says how to run the filter on an Intel GPU',
  )
  stop()
})

test('a kind left unmasked in the options reaches the model as it is, and status says so', { options: { unmaskedKinds: 'EMAIL' } }, async ($, on) => {
  const { stop } = engine(on, ['alice@x.io'], { serve: UP })
  const rows = mock.session(on)
  await $.session.start(start)

  await $.session.append(ROW)
  const { text } = await $.command.run(ccpii('status'))

  expect(JSON.stringify(rows.appended())).toContain('write to alice@x.io')
  expect(text).toContain('left unmasked: kinds EMAIL')
  stop()
})
