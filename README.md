# ccpii

**C**laude **C**ode **P**ersonally **I**dentifiable **I**nformation.

A Claude Code mod that keeps personal data away from the model. Before the model reads a prompt or a tool result, [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) runs over the text locally and each detected span is replaced with a numbered placeholder of its kind: `⟦EMAIL_0⟧`, `⟦PERSON_1⟧`, `⟦PHONE_0⟧`. The same value always gets the same placeholder for the whole session. Only text in the `⟦⟧` brackets is a placeholder, so a bare `EMAIL_0` in code or docs is left as it is and never swapped for a real value.

- A section added to the system prompt tells the model what the placeholders are and that it can use them as they are. Claude Code renders the system prompt once per conversation, so in a conversation the mod joined later (turned on mid-session, or a resumed one started without it) the same notice is added to the conversation instead, once, with the first placeholders.
- When the model calls a tool, the placeholders in the arguments are replaced with the real values just before the tool runs, and the tool's output is filtered again.
- In the transcript the user sees the real values: messages and tool rows are restored for display only; the stored conversation keeps the placeholders.
- The engine's own reminders (token budget, tool listings, todo and plan-mode reminders) are not filtered; every other attachment is, a file you mention included.
- If the filter cannot check something, the prompt is not sent and tool output is withheld from the model (fail-closed), with the reason and what to do in the status line and in the withheld text.

Placeholder kinds: `ACCOUNT`, `ADDRESS`, `DATE`, `EMAIL`, `PERSON`, `PHONE`, `URL`, `SECRET`.

## Install

From [agent-plugins](https://github.com/xchacha20-poly1305/agent-plugins):

```shell
claude plugin marketplace add https://github.com/xchacha20-poly1305/agent-plugins.git
claude plugin install ccpii@anrong-plugins
```

## Setup

Needs [`uv`](https://docs.astral.sh/uv/) on `PATH`. Then, in a session:

```
/ccpii install
```

It installs the Python dependencies into uv's cache and downloads the model's int8 ONNX export with its config and Viterbi calibration (about 1.6 GB) to `~/.cache/ccpii/privacy-filter`. Until it is installed everything is withheld; `/ccpii pause` lets content through meanwhile. The same install from a shell:

```
uv run --script server/pii_server.py --download
```

### On an Intel GPU

On an Intel GPU with XMX matrix engines (Arc, including the Arc 130T/140T in Core Ultra 200H), the filter can run its forward pass there, about five times as fast on long texts (a 15 KB file: 0.75 s against 3.8 s). ccpii does not make the GPU model: the model's ONNX exports use onnxruntime-only operators, and the conversion to OpenVINO depends on exact library versions. With an OpenVINO model at `~/.cache/ccpii/privacy-filter/openvino-int8/model.xml`, the sidecar uses the GPU, keeps the CPU model loaded beside it, and gives each GPU call 5 s before the CPU answers instead: with system memory short, the i915 driver has been seen to hang the GPU's compute work. [GPU.md](GPU.md) describes what the model must be, how the fallback works, and how one was converted. `/ccpii status` says which device answers and how often the GPU has fallen back.

## Another model or backend

Set the **Detection endpoint** option (`/config` › ccpii) to the URL of any service that answers ccpii's small HTTP protocol, and ccpii uses it in place of the bundled filter. [BACKENDS.md](BACKENDS.md) describes the protocol and ships two ready-made endpoints: any Hugging Face token-classification model, and Microsoft Presidio.

## `/ccpii`

It runs at once, mid-turn too, so it works even while the filter holds everything back.

| Command | What it does |
| --- | --- |
| `/ccpii status` | What the filter is doing, how many placeholders there are |
| `/ccpii install` | Download the model |
| `/ccpii restart` | Start the filter process again |
| `/ccpii pause` | Let content reach the model unfiltered, until `/ccpii resume` or a reload |
| `/ccpii resume` | Filter again |

## How it runs

Each session starts one sidecar process (`server/pii_server.py`) serving `POST /detect` on a Unix socket in a private directory under `/tmp`. The forward pass runs on onnxruntime (CPU); tokenizing, windows, Viterbi decoding and character offsets are opf's own code, pinned to one commit. It takes about 2 s to load, about 2.2 GB of memory, and roughly 1 s per 4 KB of text. Claude Code gives each request 30 s, so a long text goes in overlapping pieces, at most 16,384 UTF-16 code units a request, one request at a time. A sidecar that was ready and then dies is started again by the next request; one that dies before it is ready is not retried, and `/ccpii restart` starts it. Either way the directory it left behind is removed at the next start, after a reload of the mod too.

The placeholder table of each session is saved to `~/.claude/ccpii/v2/<session-id>.json` (directory mode 700), so a resumed session can still restore its placeholders.

## Files

- `hooks/register.ts`: the hooks and `/ccpii`.
- `hooks/vault.ts`: the placeholder table (masking and restoring).
- `hooks/filter.ts`: runs the sidecar and calls it.
- `server/pii_server.py`: the sidecar and the install.
- `server/gpu_worker.py`: runs a GPU model (GPU.md), a child of the sidecar.
- `examples/`: detection endpoints for other backends (BACKENDS.md).

## Limits

- Permission prompts show the arguments with placeholders, and permission rules match against them.
- A real value the model writes on its own is not filtered: only what the model reads is.
- The model can miss or over-detect spans: in testing it took `IBAN` and random path names for people, and a user name in `ls -l` output for a person. A hex digest it takes for an account number (a git commit, a sha256) is left alone. It is a data-minimisation aid, not an anonymisation guarantee.

## License

MIT
