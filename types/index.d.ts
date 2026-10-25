/**
 * The placeholder table of one session: which real value each placeholder
 * stands for, and the next number per kind.
 */
export type Vault = {
  /** The session the table belongs to; '' before any session has loaded it. */
  sessionId: string
  /** The next free number per placeholder kind (`EMAIL` → 2 after EMAIL_1). */
  counters: Record<string, number>
  /** Real value → placeholder. */
  byValue: Record<string, string>
  /** Placeholder → real value. */
  byPlaceholder: Record<string, string>
}

declare module 'claude-code' {
  interface PluginState {
    ccpii: {
      vault: Shaped<Vault>
      /** The directory of the sidecar this session runs, so a reloaded module can remove a dead one's. */
      sidecarDir: string
    }
  }
}
