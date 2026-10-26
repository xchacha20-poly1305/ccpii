import { expect, test } from 'claude-code/testing'

import { describePolicy, maskedSpans, policyOf } from './policy'
import type { Span } from './vault'

/** The text of each span the policy masks. */
function masked(kinds: string, words: string, text: string, found: [string, string][]): string[] {
  const spans: Span[] = found.map(([needle, label]) => {
    const start = text.indexOf(needle)

    return { label, start, end: start + needle.length }
  })

  return maskedSpans(policyOf(kinds, words), text, spans).map(span => text.slice(span.start, span.end))
}

test('a kind left unmasked, as a kind or a label in any case, is not masked', async () => {
  const text = 'Alice wrote from home, see example.com/a'
  const found: [string, string][] = [
    ['Alice', 'private_person'],
    ['example.com/a', 'private_url'],
  ]

  expect(masked('', '', text, found)).toEqual(['Alice', 'example.com/a'])
  expect(masked('person', '', text, found)).toEqual(['example.com/a'])
  expect(masked(' PERSON , private_url ', '', text, found)).toEqual([])
  expect(masked('PER', '', text, [['Alice', 'PER']])).toEqual([])
})

test('a span holding a word left unmasked is not masked; a word running on is no match', async () => {
  const text = '-rw-r--r-- 1 anrong anrong, Joanna, Charles Darwin, 张伟明天请假'
  const found: [string, string][] = [
    ['-rw-r--r-- 1 anrong anrong', 'private_person'],
    ['Joanna', 'private_person'],
    ['Charles Darwin', 'private_person'],
    ['张伟明天请假', 'private_person'],
  ]

  expect(masked('', 'Anrong, ann, darwin, 张伟', text, found)).toEqual(['Joanna'])
})

test('a hex digest taken for an account number is left alone; a hex secret is not', async () => {
  const text = 'commit 3f9a2c1d4e5b, key 9f86d081884c7d65, IBAN GB29NWBK60161331926819'

  expect(
    masked('', '', text, [
      ['3f9a2c1d4e5b', 'account_number'],
      ['9f86d081884c7d65', 'secret'],
      ['GB29NWBK60161331926819', 'account_number'],
    ]),
  ).toEqual(['9f86d081884c7d65', 'GB29NWBK60161331926819'])
})

test('a file path taken for a URL is left alone; a URL is not', async () => {
  const paths = ['/home/a/x.ts', '~/.ssh/config', './src/a.rs', '../b', '.claude/x.json', 'hooks/vault.ts', 'C:\\Users\\x']
  const urls = [
    'https://example.com/a',
    'www.example.org/posts/1',
    'api.internal.corp/v1/users',
    'github.com:foo/bar.git',
    'localhost:8080/admin',
    's3://bucket/key',
    '//cdn.example.com/x.js',
    'example.com',
  ]
  for (const path of paths) expect(masked('', '', path, [[path, 'private_url']])).toEqual([])
  for (const url of urls) expect(masked('', '', url, [[url, 'private_url']])).toEqual([url])
})

test('a file mode taken for a person is left alone', async () => {
  const text = 'drwxr-xr-x -rw-r--r-- lrwxrwxrwx Mrs-Robins'

  expect(
    masked('', '', text, [
      ['drwxr-xr-x', 'private_person'],
      ['-rw-r--r--', 'private_person'],
      ['lrwxrwxrwx', 'private_person'],
      ['Mrs-Robins', 'private_person'],
    ]),
  ).toEqual(['Mrs-Robins'])
})

test('status names what is left unmasked, and nothing when nothing is', async () => {
  expect(describePolicy(policyOf('', ' , '))).toBeUndefined()
  expect(describePolicy(policyOf('person,URL', 'Anrong'))).toBe('kinds PERSON, URL; words anrong')
})
