import { atom, read, update } from 'claude-code'
import type { Args, EngineInterface, Register } from 'claude-code'

import type { Vault } from '../types'
import {
  configureFilter,
  describeFilter,
  describeGpu,
  detect,
  filterStatus,
  installFilter,
  startFilter,
} from './filter'
import { CLOSE, KINDS, OPEN, emptyVault, mask, placeholderOf, unmask, unmaskDeep } from './vault'

type Blocks = Args<'session.append'>['message']['content']

// Change the tag when Vault's fields change: it declines what an earlier
// version stored, which outlives a reload.
const vaultAtom = atom({ plugin: 'ccpii', key: 'vault' } as const, emptyVault(''), {
  shape: 'vault-2',
})

/**
 * Attachments that are the engine's own prose (a reminder, a listing, a mode
 * change), never the person's data: masking them garbles what the engine
 * tells the model. Every other attachment is masked, one whose name is new
 * included, since a file the person mentioned is an attachment too.
 */
const ENGINE_PROSE = new Set([
  'total_tokens_reminder',
  'deferred_tools_delta',
  'todo_reminder',
  'plan_mode',
  'plan_mode_reentry',
  'plan_mode_exit',
])

const WITHHELD = '[ccpii] Withheld:'

const NOTICE = {
  id: 'ccpii:notice',
  scope: 'session',
  text: `# Privacy placeholders (ccpii)

A privacy filter runs in this session. Before you read them, the user's messages and every tool result have personal or secret data replaced by placeholders of the form ${OPEN}KIND_N${CLOSE}, KIND being what was detected: usually one of ${[
    ...new Set(Object.values(KINDS)),
  ].join(', ')}, or another detector's label (for example ${placeholderOf('EMAIL', 0)}, ${placeholderOf('PERSON', 1)}). Only text in these brackets is a placeholder: a bare word that looks like one is just text. The same placeholder always stands for the same real value for the whole session, and different values get different numbers.

- Treat each placeholder as the value it stands for. Write it verbatim wherever you need that value, including inside tool call arguments (file paths, shell commands, file contents, edit strings): the placeholders are replaced with the real values just before a tool runs, and the tool's output is filtered again before you see it.
- Do not guess, reconstruct or ask for the real values, and do not invent placeholders that you have not seen.
- In replies to the user, use the placeholders as they are; the user's screen shows the real values in their place.
- Content that starts with "${WITHHELD}" was held back because the filter could not check it; tell the user the reason it gives instead of guessing what it held.`,
} as const

const USAGE = `/ccpii status    what the filter is doing
/ccpii install   download the privacy-filter model (about 1.6 GB)
/ccpii restart   start the filter process again
/ccpii pause     let content reach the model unfiltered until /ccpii resume or a reload
/ccpii resume    filter again`

/** This session's table, as loaded or last changed; reset by a reload, restored from `$.state`. */
let live: Vault | undefined
/** Kept in the module alone, so a reload always filters again. */
let isPaused = false
let saving: Promise<void> = Promise.resolve()
let isSaveDirMade = false
/** The conversation whose notice this module has claimed, and whether it was this module's to add. */
let noticeClaim: { sessionId: string; isNew: Promise<boolean> } | undefined

export const register: Register = (on, options) => {
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    await $.command.register({
      name: 'ccpii',
      description: 'Privacy filter: status, install, restart, pause, resume',
      argumentHint: '[status|install|restart|pause|resume]',
      // Runs at once, mid-turn too: the way out when the filter holds everything back.
      immediate: true,
    })
    // Fire and forget: the status line is a courtesy, and the hook that set
    // this up may be long gone, its session with it, by the next change.
    configureFilter(() => void showStatus($).catch(() => {}), String(options.endpoint ?? ''))
    await startFilter($)

    return started
  })

  on('command.run', { command: 'ccpii' }, async ($, e) => ({ text: await runCommand($, e.args.trim()) }))

  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    // Rendered before the main conversation's first answer, this is the
    // system prompt the conversation keeps; a subagent's comes later.
    if (!(await $.session.messages()).some(message => message.role === 'assistant')) {
      await markNotice($, 'system prompt')
    }

    return { sections: [...composed.sections, NOTICE] }
  })

  // A slash command's text is not rewritable here; its expansion reaches the
  // model as a `command` row, which `session.append` masks.
  on('prompt.submit', async ($, e, next) => {
    if (isPaused || e.command !== undefined || e.isMeta) return next(e)
    const texts = [e.text, ...(e.context ?? [])]
    let masked: string[]
    try {
      masked = await maskTexts($, texts)
    } catch (error) {
      $.ui.toast(`Prompt not sent: ${reasonOf(error)}`)

      return { drop: `ccpii: ${reasonOf(error)}` }
    }
    const [text = e.text, ...context] = masked
    if (masked.some((one, i) => one !== texts[i]) && (await claimNotice($))) {
      context.push(NOTICE.text)
    }

    return next(context.length === 0 ? { ...e, text } : { ...e, text, context })
  }).catch(($, e, next) =>
    next.called ? next(e) : { drop: 'ccpii: the privacy filter could not check the prompt' },
  )

  // Every row the model reads is masked here: tool results, command
  // expansions, attachments, hook context, deliveries. A prompt already masked
  // at `prompt.submit` comes through unchanged, as its placeholders are skipped.
  on('session.append', async ($, e, next) => {
    if (isPaused || !isReadByModel(e) || isOwnNote($, e)) return next(e)
    let content: Blocks
    let isChanged: boolean
    try {
      ;({ content, isChanged } = await maskBlocks($, e.message.content))
    } catch (error) {
      content = withheld(e.message.content, reasonOf(error))
      isChanged = true
    }
    const stored = await next({ ...e, message: { ...e.message, content } })
    // The first placeholders (or withheld text) of a conversation whose system
    // prompt has no notice: the notice follows them in, once. A subagent's
    // prompt is rendered when it starts, and has it.
    if (e.agentId === undefined && isChanged && (await claimNotice($))) {
      await $.session.append({ message: { type: 'user', content: [{ type: 'text', text: NOTICE.text }] } })
    }

    return stored
  }).catch(($, e, next) => {
    if (next.called || isPaused || !isReadByModel(e)) return next(e)

    return next({ ...e, message: { ...e.message, content: withheld(e.message.content, 'the filter hook failed') } })
  })

  on('tool.call', async ($, e, next) => next(unmaskDeep(await currentVault($), e))).catch(
    ($, e, next) =>
      next.called ? next(e) : { deny: 'ccpii: the placeholders in this call could not be restored.' },
  )

  on('ui.render', { component: 'UserMessage' }, async ($, e, next) => {
    const vault = await read($, vaultAtom)

    return next({ ...e, props: { ...e.props, text: unmask(vault, e.props.text) } })
  })

  on('ui.render', { component: 'AssistantMessage' }, async ($, e, next) => {
    const vault = await read($, vaultAtom)

    return next({ ...e, props: { ...e.props, text: unmask(vault, e.props.text) } })
  })

  on('ui.render', { component: 'ToolUse' }, async ($, e, next) => {
    const vault = await read($, vaultAtom)
    const input = unmaskDeep(vault, e.props.input)

    return input === e.props.input ? next(e) : next({ ...e, props: { ...e.props, input } })
  })
}

async function runCommand($: EngineInterface, args: string): Promise<string> {
  switch (args) {
    case '':
    case 'status': {
      const vault = await currentVault($)
      const lines = [
        `filter: ${describeFilter()}`,
        ...(filterStatus() === 'external' ? [] : [`device: ${describeGpu()}`]),
        `filtering: ${isPaused ? 'PAUSED, content reaches the model unfiltered' : 'on'}`,
        `placeholders: ${Object.keys(vault.byPlaceholder).length}, saved in ${await savePath($, vault.sessionId)}`,
      ]

      return [...lines, '', USAGE].join('\n')
    }
    case 'install':
      if (filterStatus() === 'external') return `Nothing to install: ${describeFilter()}.`
      await installFilter($)

      return `${describeFilter()}. The status line shows when it is done.`
    case 'restart':
      await startFilter($)

      return `${describeFilter()}.`
    case 'pause':
      isPaused = true
      await showStatus($)

      return 'Filtering paused. Prompts and tool results now reach the model unfiltered, until /ccpii resume or a reload. Placeholders already issued are still restored in tool calls.'
    case 'resume':
      isPaused = false
      await showStatus($)

      return `Filtering on; ${describeFilter()}.`
    default:
      return `Unknown argument "${args}".\n\n${USAGE}`
  }
}

/**
 * Whether this conversation's model has been told about the placeholders,
 * kept in `$.store` by session id so a resumed conversation knows. The
 * system prompt is rendered once per conversation (--system-prompt-snapshot),
 * so a mod loaded after the first request adds its section to nothing; the
 * notice then goes into the conversation itself.
 */
async function markNotice($: EngineInterface, where: 'system prompt' | 'conversation'): Promise<void> {
  await $.store.set(`notice:${await $.session.id()}`, where)
}

/** True once per conversation that has no notice yet, and records that it now gets one. */
async function claimNotice($: EngineInterface): Promise<boolean> {
  const sessionId = await $.session.id()
  if (noticeClaim?.sessionId === sessionId) return false
  // Claimed before the next await, so two rows masked at once add one notice.
  const isNew = (async () => {
    if ((await $.store.get(`notice:${sessionId}`)) !== undefined) return false
    await markNotice($, 'conversation')

    return true
  })()
  noticeClaim = { sessionId, isNew }

  return isNew
}

/** The notice this module appended, which passes as it is. */
function isOwnNote($: EngineInterface, e: Args<'session.append'>): boolean {
  return e.door === 'note' && e.origin.kind === 'plugin' && 'name' in e.origin && e.origin.name === $.plugin.name
}

/** Every row the model reads that may carry data: not its own, not a notice, not the engine's own prose. */
function isReadByModel(e: Args<'session.append'>): boolean {
  const { type, name } = e.message
  if (type === 'attachment') return name === undefined || !ENGINE_PROSE.has(name)

  return type !== 'assistant' && type !== 'system'
}

/** Masks each text against one table, issuing placeholders as it goes. */
async function maskTexts($: EngineInterface, texts: readonly string[]): Promise<string[]> {
  if (texts.every(text => text.trim() === '')) return [...texts]
  const spans = await detect($, texts)
  await currentVault($)

  // From here to the end nothing awaits, so no other hook changes `live`
  // between reading it and writing it back.
  const before = live ?? emptyVault('')
  let vault = before
  const masked = texts.map((text, i) => {
    const result = mask(vault, text, spans[i] ?? [])
    vault = result.vault

    return result.text
  })
  if (vault !== before) {
    live = vault
    saving = saving.then(() => save($)).catch(error => $.ui.log(`ccpii: saving failed: ${error}`, { to: 'debug' }))
  }

  return masked
}

/** `content` with its texts masked, and whether any of them changed. */
async function maskBlocks($: EngineInterface, content: Blocks): Promise<{ content: Blocks; isChanged: boolean }> {
  const texts: string[] = []
  mapTexts(content, text => (texts.push(text), text))
  if (texts.length === 0) return { content, isChanged: false }
  const masked = await maskTexts($, texts)
  if (masked.every((text, i) => text === texts[i])) return { content, isChanged: false }
  let i = 0

  return { content: mapTexts(content, text => masked[i++] ?? text), isChanged: true }
}

function withheld(content: Blocks, reason: string): Blocks {
  return mapTexts(content, () => `${WITHHELD} ${reason}.`).map(block =>
    block.type === 'tool_result' ? { ...block, is_error: true } : block,
  )
}

/**
 * Rewrites every text the model reads in `content`, in document order: text
 * blocks and tool results, a tool result's content being a string or blocks.
 */
function mapTexts(content: Blocks, fn: (text: string) => string): Blocks {
  return content.map(block => {
    if (block.type === 'text' && typeof block.text === 'string') return { ...block, text: fn(block.text) }
    if (block.type !== 'tool_result') return block
    if (typeof block.content === 'string') return { ...block, content: fn(block.content) }
    if (Array.isArray(block.content)) return { ...block, content: mapTexts(block.content as Blocks, fn) }

    return block
  })
}

function reasonOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * The current session's table: `live` when it is this session's, else what
 * `$.state` holds across a reload, else the file a resumed session left.
 */
async function currentVault($: EngineInterface): Promise<Vault> {
  const sessionId = await $.session.id()
  if (live?.sessionId === sessionId) return live

  const held = await read($, vaultAtom)
  const restored = held.sessionId === sessionId ? held : await readSaved($, sessionId)
  // Another hook may have loaded it, and issued placeholders, meanwhile.
  if (live?.sessionId !== sessionId) live = restored

  return live
}

async function readSaved($: EngineInterface, sessionId: string): Promise<Vault> {
  const path = await savePath($, sessionId)
  if (!(await $.fs.exists(path))) return emptyVault(sessionId)

  return { ...(JSON.parse(await $.fs.read(path)) as Vault), sessionId }
}

async function save($: EngineInterface): Promise<void> {
  const vault = live
  if (vault === undefined) return
  await update($, vaultAtom, () => vault)
  const path = await savePath($, vault.sessionId)
  if (!isSaveDirMade) {
    // Only the owner may read the real values the table holds.
    await $.process.run(['mkdir', '-p', '-m', '700', path.slice(0, path.lastIndexOf('/'))])
    isSaveDirMade = true
  }
  await $.fs.write(path, JSON.stringify(vault))
  await showStatus($)
}

async function savePath($: EngineInterface, sessionId: string): Promise<string> {
  const home = await $.env.get('HOME')

  // v2: the bracketed placeholders; a table of the bare ones is not read.
  return `${home}/.claude/ccpii/v2/${sessionId}.json`
}

async function showStatus($: EngineInterface): Promise<void> {
  if (isPaused) {
    $.ui.status('PAUSED, content reaches the model unfiltered')
  } else if (filterStatus() === 'ready' || filterStatus() === 'external') {
    const vault = live ?? (await read($, vaultAtom))
    $.ui.status(`${Object.keys(vault.byPlaceholder).length} placeholders`)
  } else {
    $.ui.status(describeFilter())
  }
}
