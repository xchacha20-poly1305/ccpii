# Using another detection backend

By default ccpii runs [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) in a sidecar of its own. To use a different model, a language the default handles poorly, or a detection service you already run, point ccpii at an HTTP endpoint that speaks the small protocol below. ccpii then starts no sidecar and sends every check there.

## Setting it

In a session, open `/config` and set **ccpii › Detection endpoint** to the endpoint's URL, for example `http://127.0.0.1:8765/detect`. The mod reloads with it at once. Or in `~/.claude/settings.json`:

```json
{
  "pluginConfigs": {
    "ccpii": { "options": { "endpoint": "http://127.0.0.1:8765/detect" } }
  }
}
```

Clear the field to go back to the bundled filter. `/ccpii status` says which is in use; `/ccpii install` and `/ccpii restart` do nothing for an endpoint, since you run it yourself.

## The protocol

ccpii sends `POST <endpoint>` with a JSON body, one entry per text to check:

```json
{ "texts": ["Contact Maria Garcia at maria@example.com", "no data here"] }
```

The endpoint answers `200` with one list of spans per text, in the same order:

```json
{
  "spans": [
    [
      { "label": "PERSON", "start": 8, "end": 20 },
      { "label": "EMAIL_ADDRESS", "start": 24, "end": 41 }
    ],
    []
  ]
}
```

- **Offsets** are UTF-16 code units, start inclusive, end exclusive: what JavaScript's `String.slice` takes. Python counts code points, so convert: `len(text[:i].encode("utf-16-le")) // 2`. The two differ only past characters outside the Basic Multilingual Plane, such as emoji.
- **Labels** become the placeholder kind. openai/privacy-filter's labels map to short kinds (`private_email` → `EMAIL`); any other label is upper-cased with characters outside `A-Z`, `0-9` and `_` turned into `_`, so `EMAIL_ADDRESS` gives `⟦EMAIL_ADDRESS_0⟧` and `per` gives `⟦PER_0⟧`. An empty label gives `PII`.
- **Overlaps** are allowed: where spans overlap, the one that starts first (the longer, at the same start) is kept.
- **Size and time**: a request carries at most 16,384 UTF-16 code units of text in all, and the endpoint has 30 seconds to answer it; Claude Code ends a request after that, and the text is withheld. A longer text arrives in overlapping pieces, each its own entry in `texts`, and ccpii puts the spans back together. Requests come one at a time.
- **Failures** are fail-closed. A non-2xx status, a body that is not JSON, a list count that differs from the texts, or a span outside its text: the text is not sent to the model. The reason is shown in the status line and in what the model reads in its place.

The endpoint gets the real data. Bind it to loopback (`127.0.0.1`); an endpoint on another machine sends the data there, and ccpii sends no credentials.

## Ready-made endpoints

Both are single [uv](https://docs.astral.sh/uv/) scripts in `examples/`, installing what they need on first run and serving on loopback.

| Script | Backend | Run it with |
| --- | --- | --- |
| `examples/hf_ner_server.py` | Any Hugging Face token-classification model, through `transformers` | `uv run --script examples/hf_ner_server.py --model dslim/distilbert-NER --port 8765` |
| `examples/presidio_server.py` | [Microsoft Presidio](https://microsoft.github.io/presidio/) with spaCy's small English model | `uv run --script examples/presidio_server.py --port 8765` |

For `hf_ner_server.py`, `--model` takes any model id or local path with a token-classification head: a multilingual NER model, a fine-tune of your own. `--labels PER,ORG` keeps only those entity groups. Its labels are the model's entity groups (`PER`, `ORG`, `LOC`, `MISC` for the model above), so the placeholders read `⟦PER_0⟧`.

For `presidio_server.py`, `--threshold` sets the lowest score kept (0.5 by default). Change `NLP_CONFIG` in the script for another language or spaCy model.

## Writing your own

Anything that answers the protocol works: a wrapper around a cloud DLP API, a set of rules of your own, a different runtime. Take one of the examples as the skeleton and replace its `detect(text)`, which returns the spans of one text with Python offsets converted to UTF-16.

Check an endpoint before pointing ccpii at it:

```sh
curl -s -X POST --data '{"texts": ["Contact Maria Garcia at maria@example.com"]}' \
  http://127.0.0.1:8765/detect
```

ccpii does not start or stop the endpoint. To keep one running, run it as a user service (`systemd --user`, launchd) or in a terminal of its own.
