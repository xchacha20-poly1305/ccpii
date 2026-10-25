import { expect, test } from 'claude-code/testing'

import { emptyVault, kindOf, mask, placeholderOf as ph, unmask, unmaskDeep } from './vault'
import type { Span } from './vault'

// Placeholders are spelled through `ph` throughout: a literal one in this
// file would be restored by a running ccpii while the file is edited.

/** The spans of each `needle` in `text`, as the sidecar reports them. */
const spansOf = (text: string, found: Record<string, string>): Span[] =>
  Object.entries(found).map(([needle, label]) => {
    const start = text.indexOf(needle)

    return { label, start, end: start + needle.length }
  })

test('the same value gets the same placeholder, a new one the next number', async () => {
  const text = 'mail a@x.io, then b@x.io, then a@x.io again'
  const spans: Span[] = [
    { label: 'private_email', start: 5, end: 11 },
    { label: 'private_email', start: 18, end: 24 },
    { label: 'private_email', start: 31, end: 37 },
  ]
  const { vault, text: masked } = mask(emptyVault('s'), text, spans)

  expect(masked).toBe(`mail ${ph('EMAIL', 0)}, then ${ph('EMAIL', 1)}, then ${ph('EMAIL', 0)} again`)
  expect(vault.counters).toEqual({ EMAIL: 2 })
  expect(unmask(vault, masked)).toBe(text)
})

test('each kind numbers on its own, across texts', async () => {
  const first = 'Alice Smith <alice@x.io>'
  const one = mask(emptyVault('s'), first, spansOf(first, { 'Alice Smith': 'private_person', 'alice@x.io': 'private_email' }))
  const second = 'Bob Ray wrote to alice@x.io'
  const two = mask(one.vault, second, spansOf(second, { 'Bob Ray': 'private_person', 'alice@x.io': 'private_email' }))

  expect(one.text).toBe(`${ph('PERSON', 0)} <${ph('EMAIL', 0)}>`)
  expect(two.text).toBe(`${ph('PERSON', 1)} wrote to ${ph('EMAIL', 0)}`)
})

test('a text with nothing new keeps the vault object itself', async () => {
  const vault = emptyVault('s')

  expect(mask(vault, 'nothing here', []).vault).toBe(vault)
})

test('a span over a placeholder already issued is not masked again', async () => {
  const { vault } = mask(emptyVault('s'), 'a@x.io', [{ label: 'private_email', start: 0, end: 6 }])
  const text = `to ${ph('EMAIL', 0)}, ok`
  const again = mask(vault, text, [{ label: 'private_email', start: 3, end: 3 + ph('EMAIL', 0).length }])

  expect(again.text).toBe(text)
  expect(again.vault).toBe(vault)
})

test('overlapping spans keep the earlier, longer one', async () => {
  const text = 'call +1 555 0100 now'
  const { text: masked } = mask(emptyVault('s'), text, [
    { label: 'private_phone', start: 5, end: 16 },
    { label: 'account_number', start: 8, end: 16 },
  ])

  expect(masked).toBe(`call ${ph('PHONE', 0)} now`)
})

test('only bracketed placeholders the vault issued are restored; bare names stay', async () => {
  const { vault } = mask(emptyVault('s'), 'a@x.io b@x.io', [
    { label: 'private_email', start: 0, end: 6 },
    { label: 'private_email', start: 7, end: 13 },
  ])
  const bare = ['EMAIL', '1'].join('_')

  expect(unmask(vault, `${ph('EMAIL', 1)} ${bare} ${ph('EMAIL', 10)} ⟦x ${ph('EMAIL', 0)}`)).toBe(
    `b@x.io ${bare} ${ph('EMAIL', 10)} ⟦x a@x.io`,
  )
})

test('unmaskDeep restores strings inside tool input and keeps untouched parts', async () => {
  const { vault } = mask(emptyVault('s'), 'a@x.io', [{ label: 'private_email', start: 0, end: 6 }])
  const input = { tool: 'Bash', command: `echo ${ph('EMAIL', 0)}`, nested: { list: [ph('EMAIL', 0), 'plain'], n: 3 } }
  const restored = unmaskDeep(vault, input)

  expect(restored).toEqual({ tool: 'Bash', command: 'echo a@x.io', nested: { list: ['a@x.io', 'plain'], n: 3 } })
  const untouched = { a: 'plain', b: [1, 2] }
  expect(unmaskDeep(vault, untouched)).toBe(untouched)
})

test('UTF-16 offsets past Chinese text and emoji slice correctly', async () => {
  const text = '联系 😀 张三：zhang@x.cn'
  const start = text.indexOf('zhang@x.cn')
  const { text: masked } = mask(emptyVault('s'), text, [
    { label: 'private_person', start: text.indexOf('张三'), end: text.indexOf('张三') + 2 },
    { label: 'private_email', start, end: start + 'zhang@x.cn'.length },
  ])

  expect(masked).toBe(`联系 😀 ${ph('PERSON', 0)}：${ph('EMAIL', 0)}`)
})

test('a hex digest taken for an account number is left alone; a hex secret is not', async () => {
  const text = 'commit 3f9a2c1d4e5b, key 9f86d081884c7d65, IBAN GB29NWBK60161331926819'
  const at = (needle: string) => text.indexOf(needle)
  const { text: masked } = mask(emptyVault('s'), text, [
    { label: 'account_number', start: at('3f9a2c1d4e5b'), end: at('3f9a2c1d4e5b') + 12 },
    { label: 'secret', start: at('9f86d081884c7d65'), end: at('9f86d081884c7d65') + 16 },
    { label: 'account_number', start: at('GB29'), end: text.length },
  ])

  expect(masked).toBe(`commit 3f9a2c1d4e5b, key ${ph('SECRET', 0)}, IBAN ${ph('ACCOUNT', 0)}`)
})

test('a label of another backend becomes its own kind', async () => {
  expect(kindOf('private_email')).toBe('EMAIL')
  expect(kindOf('EMAIL_ADDRESS')).toBe('EMAIL_ADDRESS')
  expect(kindOf('per')).toBe('PER')
  expect(kindOf('US-SSN')).toBe('US_SSN')
  expect(kindOf('')).toBe('PII')
})
