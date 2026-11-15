# radiance-ctools: Codex custom tools on upstream Radiance

radiance-ctools is a downstream fork of [Radiance](https://codeberg.org/StillDeadcode/radiance). It is upstream's
release plus one patch series: OpenAI Responses **freeform custom tools** on `/v1/responses`, the kind Codex's
`apply_patch` is, with the Responses tool behaviour around them. The series is ported from
[Strata](https://github.com/Niko1221/Strata)'s `serve/responses.py` and `serve/frontend.py` (MIT; see NOTICE). With it,
Codex works against Radiance directly, without a translating proxy.

## What the series adds

- **Custom tools** (`{"type":"custom"}`):
  - offered to the model as a function tool taking one string `input`, with a grammar `format` shown to the model
    as text;
  - the reply's call is a `custom_tool_call` item with the raw `input`, streamed as
    `response.custom_tool_call_input.delta` and `.done`;
  - `custom_tool_call` and `custom_tool_call_output` history items are served;
  - a custom tool inside a `namespace` is named `<namespace>.<name>`, as in Strata.
- **Codex's `additional_tools` input items:** their tools join the request's own.
- **Tool results put in call order**, since a chat template places results by position.
- **A `tool_choice` the request cannot honour** acts as `"auto"` rather than failing the request.
- **Codex's local compaction** (with a provider that is not OpenAI) is an ordinary request, served as one.

Function tools, namespaced function tools and `/v1/chat/completions` behave exactly as upstream's. The patch does not
change how function tools are rendered in the prompt.

The full behaviour, and **the list of every difference from Strata** (each kept because upstream serves function
tools that way), are in docs/GUIDE.md, section 5.9: the paragraphs on freeform custom tools, and "The Responses tool
path follows Strata's ...".

## Where it lives

- **Logic, in its own headers:** `core/server/oai_custom_tools.h`, `oai_additional_tools.h`, `oai_tool_results.h` and
  `oai_tool_choice.h`. `oai.cpp`, `oai.h` and `server.cpp` carry thin hooks only, so a rebase touches few lines.
- **Tests:**
  - `tests/custom_tools_test.cpp` and `tests/additional_tools_test.cpp`;
  - Strata compatibility: `the_same_requests_answer_as_strata_does` compares 30 cases with answers recorded from
    Strata's own server (`tests/data/chat/strata-compat-custom-tools.json`, re-recordable with
    `tests/data/chat/strata_compat_gen.py`);
  - function-call regression: goldens recorded on the unmodified upstream release.

## Branches and releases

| ref | what |
|---|---|
| `ctools/<ver>` | upstream's tag `v<ver>` plus the series, as discrete commits (never squashed, so a rebase stays readable) |
| `main` | the `ctools/<ver>` that is the current release of this fork |

The series is `git log v<ver>..ctools/<ver>`. Images are tagged `radiance:<ver>-ctools` and
`radiance:<ver>-ctools-<sha>`, never `latest`. Their labels carry upstream's version and the fork's commit.

## How it follows upstream

A scheduled job on the maintainers' build host watches upstream's release tags and the published image labels. For
each new release it:

1. rebases the series onto the release's commit as `ctools/<ver>`;
2. runs the host test suite (the Strata comparison included) inside `docker/build.sh`'s test stage;
3. builds a candidate image, for a release of the same major version.

The job never pushes and never deploys:

- **A conflict or a failed test:** it stops, and a maintainer resolves it by hand.
- **A new major version** (2.0 changed the server's configuration, removed CLI flags and dropped the fp8 KV cache):
  the job only rebases and tests, and flags it for a migration. No image is built automatically.

## Promotion

A candidate becomes a release here only after a maintainer has checked it on real hardware:

- the tool-call benchmark;
- prompt and decode speed against the previous release;
- a real Codex session editing files with `apply_patch`, over several turns.

Then `ctools/<ver>` is pushed, and `main` moves to it.

## Resolving a rebase conflict

```sh
git switch -c ctools/<ver> ctools/<prev>
git rebase --onto v<ver> v<prev>
# resolve each stop keeping the commit's intent and the hooks thin, then: git rebase --continue
# build, and run custom_tools_test, additional_tools_test and server_test
```

If upstream changed behaviour that the series ports from Strata, re-record Strata's answers with
`tests/data/chat/strata_compat_gen.py` (run inside a Strata checkout) and check the comparison.
