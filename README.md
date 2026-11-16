# NEET counselling RAG + college predictor (CGPR take-home)

A private, locally hosted RAG assistant for **NEET.info**-style counselling Q&A (core task) plus a
**Track A** college predictor. Everything runs on Ollama with an open-weight model; nothing is sent to a
cloud LLM API. Data is **synthetic**: every college, rank and seat count was invented by `data/make_data.py`.
Nothing here describes a real institution.

## What it does
- `ask`: answers cut-off questions from the indexed table + notes **with citations** (`[R003]`, `[N2]`), and
  **refuses** ("Not in the sources") when the data does not support an answer.
- `predict`: from rank + category (+ optional state / type / course / year) lists plausible colleges, each with
  the reason and the source row it came from. Eligibility is computed in plain code; the LLM never does it.
- Refusal is enforced in code, not left to a small model:
  1. injection filter ("ignore the documents...") blocks and logs the question;
  2. unknown-term gate: a college name or year that appears nowhere in the sources is refused;
  3. similarity gate: nothing close enough in the index is refused;
  4. aggregate guard: "lowest / highest / compare / which college ..." questions are refused (the model only sees a
     handful of rows, so it cannot rank or total a table); `predict` is the tool for that;
  5. verifier: the model must cite retrieved rows and quote them; the cited row must match the college, course,
     category and year the question names (so "Rajghat BDS", a course it does not offer, cannot be answered from its MBBS
     row); any figure in the answer that is not in the cited row turns the answer into a refusal.
- Advice questions ("Should I...", "Will I definitely get...") get a no-guarantee reply plus options from the data.
- Every refusal / block is appended to `logs/events.jsonl` (local file).

## Run it (Linux/macOS)
Needs Python 3.12 and [Ollama](https://ollama.com).

**Machine this is meant to start on** (a recommendation; I only measured the two rows in "Hardware actually used"):
Ubuntu 22.04+ (or similar Linux), 4+ CPU cores, 16 GB RAM, 20 GB free disk.
- *GPU path (recommended):* an NVIDIA GPU with 8+ GB VRAM runs `qwen2.5:7b-instruct` (measured: 5.1 GB on a T4, ~1.6 s/case).
- *CPU fallback:* works with `qwen2.5:3b` and no GPU, but slower (~5 s/case on an M1; expect longer on a plain x86 CPU).
- *Workstation target (not run):* a 24-48 GB card for a ~32B model; see `design_note.md` section 4 for the memory estimate.
```bash
ollama serve &                               # if not already running
ollama pull nomic-embed-text                 # embeddings (274 MB)
ollama pull qwen2.5:3b                       # laptop model; use qwen2.5:7b-instruct on a 16 GB GPU
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python data/make_data.py                     # optional: data/ already ships the CSV
python neetrag.py index                      # build the index (323 chunks, seconds)
python neetrag.py ask "What was the 2025 closing rank for Sundarpur Government Medical College MBBS General category?"
python neetrag.py ask "What is the cut-off for Himalaya Institute of Medical Sciences?"   # refusal
python neetrag.py predict --rank 4000 --category General --state "Tamil Nadu"   # MBBS and BDS; add --course BDS to narrow
python test_core.py                          # offline checks
python eval.py                               # 28-case evaluation, prints a score table
```
Model is chosen with `CHAT_MODEL` (default `qwen2.5:3b`). If your shell sets `OLLAMA_MODELS` to a drive that is
not mounted, the server will not start; unset it or point it somewhere that exists.

**Docker:** see the header of `docker-compose.yml` (Ollama + app containers; CPU by default, add
`docker-compose.gpu.yml` for an NVIDIA GPU). What I actually tested, on an Apple M1 with Docker Desktop (CPU only):
the image builds, `docker compose up -d ollama` starts, models pull inside the container, the app container reaches
Ollama, and `python test_core.py` passes inside the container. What I could **not** verify: end-to-end answers in Docker.
Docker Desktop on Apple Silicon cannot use the Mac's GPU and ran `qwen2.5:3b` at ~0.3 tokens/s, so index build and
answers timed out. This is a limit of that environment (not tested on a native Linux box or with an NVIDIA GPU). On a
Mac, use the native Ollama path above. Testing also found that the first compose file hard-required an NVIDIA GPU
(it failed on machines without one); the GPU block is now a separate override file.

**Colab (free T4):** open `colab.ipynb`, set the runtime to T4, upload the project zip, run all cells.

## Hardware actually used
| Run | Hardware | Model |
|---|---|---|
| Results below | MacBook, Apple M1, 8 GB RAM, Ollama (Metal) | `qwen2.5:3b` (4-bit), `nomic-embed-text` |
| Colab check | Free Colab Tesla T4 (15 GB) | `qwen2.5:7b-instruct` (4.7 GB, 4-bit), `nomic-embed-text`; both 100% on GPU, 5.1 GB VRAM in use |

Target deployment (not run): one RTX 6000-class card, **assumed 48 GB VRAM**. See `design_note.md`.

## Evaluation (`eval_set.json`, 28 cases)
Run 1: M1 / `qwen2.5:3b`. Run 2: free Colab T4 / `qwen2.5:7b-instruct`.

| kind | M1 / 3B | Colab T4 / 7B | total |
|---|---|---|---|
| answer (cited, correct figure; 8 table lookups + 2 prose notes) | 9 | 10 | 10 |
| refuse (absent entity/year, out-of-scope, planted figure, wrong course, aggregation) | 12 | 12 | 12 |
| advice (no-guarantee reply) | 2 | 2 | 2 |
| predictor invariants | 4 | 4 | 4 |
| **all** | **27** | **28** | **28** |
| avg time per case | 4.6-7.0 s (varies with load) | 1.6 s | |

Honest limits:
- The one 3B miss (A2) was a *false refusal*: the model gave the right number but cited the neighbouring BDS row, so the
  verifier refused it. Safe failure, but it costs recall. The 7B model passed A2. One run each, temperature 0; I did not repeat runs.
- The Colab 7B run was made before two later changes (predictor now lists MBBS and BDS by default; index built in batches of 32). Those touch only deterministic code (the predictor checks pass in the final M1 run, and `test_core.py` passes); I did not re-run Colab.
- Two answer bugs were found by my own review before these final runs, and fixed + tested: a wrong-row answer ("Rajghat BDS" answered from the MBBS row, because the figure really was in the cited row) and an aggregation question ("lowest closing rank") answered from six rows. Cases R11 and R12 now cover them.
- The 7B run exposed a crash: the model returned the answer as a JSON number and the verifier called `.replace` on it.
  Fixed (answer, cites and JSON shape are now coerced/checked) and covered in `test_core.py`.
- Case R10 (lowercase unknown college name) passed in this run, but the unknown-term gate cannot catch lowercase names by
  design; it was refused by the model abstaining. Treat R10 as luck, not coverage.
- The slot check only constrains what the question names (a question that omits the course is not checked for it), and the aggregate guard is a coarse regex that also refuses harmless phrasings such as "which college ...".
- 28 cases on invented data prove the guards work, not that the system is accurate on real NEET data.
- The injection filter is a regex list. A cleverly phrased attack could slip past it; the figure verifier is the backstop.
  Documents are my own synthetic files, so indirect injection through documents is not tested here.

## Layout
```
neetrag.py       index, retrieve (BM25 + embeddings, rank fusion), guards, ask, predict, CLI
eval.py          runs eval_set.json
eval_set.json    28 cases with expected behaviour
test_core.py     offline checks for predictor, guards, verifier
data/            make_data.py -> cutoffs.csv (315 rows), notes.md (8 notes)
colab.ipynb      free-T4 run
design_note.md   architecture, model, GPU/memory, evaluation, privacy, cost (design_note.pdf = same, exported)
docker-compose.yml, docker-compose.gpu.yml   CPU compose + optional NVIDIA override
```

## AI tools used
Built with Claude Code (Anthropic) as a coding assistant, with me directing the design and reviewing the code.
Runtime models are open-weight only: Qwen2.5 (chat) and nomic-embed-text (embeddings) via Ollama.

## Data and licences
Data: synthetic, generated by `data/make_data.py` (seed 7). No public dataset and no personal data used.

Models (as listed by Ollama's `show --license`; check the model cards before any commercial use):

| Model | Used for | Licence |
|---|---|---|
| `qwen2.5:3b` | laptop dev/demo run | **Qwen Research License** (research use; not for commercial deployment) |
| `qwen2.5:7b-instruct` | Colab T4 run | Apache 2.0 |
| ~32B Qwen2.5 instruct (target, not run) | workstation | Apache 2.0 |
| `nomic-embed-text` | embeddings | Apache 2.0 |

So the 3B model is for this assignment's laptop demo only. An office deployment should use the 7B/14B/32B models.
