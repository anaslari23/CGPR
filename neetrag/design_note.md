# Design note: private RAG for NEET.info counselling + college predictor

## 1. What I built
A local assistant that answers cut-off questions from a table plus short policy notes, with citations, and refuses
when the sources do not support an answer. A Track A predictor lists college options for a rank and category, each
with a reason and a source row. Data is synthetic (invented colleges and ranks); no student records are used.

## 2. Architecture
```
question -> guards -> retrieve (BM25 + embeddings, rank fusion) -> LLM (JSON: answer, cites, quote) -> verifier -> answer | refusal
rank+category -> predictor (plain code over the CSV) -> options with reasons and row ids
```
- **Chunks:** one chunk per CSV row (`[R003] college | course | category | year | closing rank | seats`) and one per note.
  A row is the unit of citation, so a table is never cut mid-row.
- **Retrieval:** BM25 handles exact names and numbers; `nomic-embed-text` embeddings handle wording; reciprocal rank
  fusion combines them. Both indexes are in memory, built from `index/`.
- **Refusal is code, not a prompt.** A 3-4B model is easy to push into guessing, so five checks sit outside the model:
  (1) an injection filter blocks "ignore the documents" style input and logs it; (2) a college name or year that does not
  occur in the sources is refused before retrieval; (3) a minimum-similarity gate; (4) an aggregate guard that refuses "lowest/compare/which college" questions, since a few
  retrieved rows cannot support a ranking; (5) a verifier requires the cited rows to have been retrieved and to match the
  college, course, category and year the question names (otherwise "Rajghat BDS", which does not exist, was answered from
  Rajghat's MBBS row), the quote to be present in them, and every figure in the answer to appear in the cited row (row ids
  are stripped first, otherwise "1" would match "[R1]"). Tests and my own review caught each of these.
- **Predictor is deterministic.** Eligibility is `closing_rank >= your rank` for the chosen category and year. The model
  never computes it. A rank margin sets the tier (safe / likely / borderline). Output always says past ranks are not
  predictions. Unknown year or category is refused.

## 3. Model choice
Open weights, served by Ollama (llama.cpp), 4-bit quantised, JSON mode, temperature 0.
- **Tested here:** `qwen2.5:3b` on an Apple M1 with 8 GB RAM.
- **Tested on Colab (free Tesla T4, 15 GB):** `qwen2.5:7b-instruct`, 4.7 GB at 4-bit. Measured: 100% on GPU, 5.1 GB VRAM in use with the embedder loaded, 28/28 on the final eval at 1.6 s per case.
- **Target on the workstation:** a ~32B instruct model at 4-bit (Qwen2.5-32B-Instruct class). Qwen is a good fit: strong
  JSON/instruction following at this size, permissive licence. I would re-run the eval before choosing between it and
  a similar-size alternative; I have not run 32B.

## 4. GPU and memory estimate (workstation, estimated, not measured)
I **assume 48 GB VRAM** (RTX 6000 Ada class) and do not assume a card generation beyond that.
| Item | VRAM |
|---|---|
| 32B weights, 4-bit (Q4_K_M) | ~20 GB |
| KV cache: 64 layers x 8 KV heads x 128 dim x 2 (K,V) x 2 bytes = ~0.25 MB/token; 4 users x 8k tokens = 32k tokens | ~8 GB |
| Embedding model + a small reranker (optional) | ~1.5 GB |
| Runtime overhead / fragmentation | ~3 GB |
| **Total** | **~32 GB, leaving ~16 GB headroom** |

Speed: decoding is memory-bandwidth bound. At ~960 GB/s and a ~20 GB model the ceiling is ~48 tokens/s for one stream;
I would plan on 25-35 tokens/s real. Answers here are short (<100 tokens), so about 3-4 s each plus retrieval. This is
arithmetic, not a benchmark. On a 24 GB card the same design works with a 14B model.
Measured: M1/3B 4.6 s per eval case; Colab T4/7B 1.6 s per case (7B model, 5.1 GB VRAM). Scaling to 32B is an estimate: roughly 4.5x the weights, so expect a few times slower per token on a faster card.

## 5. Evaluation
28 cases in `eval_set.json`, expected behaviour fixed in advance: 10 answerable (8 table lookups with the right figure and
cited row, 2 prose-note questions), 12 refusals (unknown college, year not in data, out-of-scope fee/stipend, a planted
"cut-off is 1", a region not in the data, a lowercase fake name, a course the college does not offer, an aggregation), 2 advice cases that must not guarantee anything, 4 predictor checks (bad category,
bad year, empty result, every option actually inside the closing rank). `eval.py` prints a pass table.
Result on M1/3B: 27/28; on Colab T4/7B: 28/28. The 3B miss was a false refusal (right number, wrong cited row, so the verifier rejected it): safe,
but it lowers recall. The 7B run also caught a verifier crash on a numeric JSON answer, now fixed and tested. Limits: tiny, invented data; regex injection filter; the unknown-term gate misses lowercase names
(R10 passed because the model abstained, not because a guard caught it). These numbers show the guards work, not that
accuracy on real data is known.

## 6. Privacy and data handling
- **Where things live:** index (`index/`) and logs (`logs/events.jsonl`) are plain files on the machine running the app.
  The app talks only to Ollama on `localhost` (or the compose network). No cloud LLM API is used anywhere.
- **Off-machine:** nothing in the main path. For my own testing I used a free Colab T4, so the synthetic data and queries
  went to Google's servers. That was acceptable only because the data is synthetic.
- **On-site changes:** run on the workstation with no internet needed after model download; separate index directory,
  log file and Ollama instance (or OS user) per business, so NEET, Hirrx and editorial data are never in one index;
  disk encryption and file permissions on the index; log retention limit; no raw student rows in logs (logs hold the
  question, status and reason). Smart Catalyst and NEET.info stay separate indexes, matching the brief.
- Staff stay responsible for any decision about a person; the predictor says so in every output.

## 7. Cost
- **Experiment:** free Colab tier + a laptop: no spend.
- **On-site:** workstation GPU is already owned. At ~300 W under load, 8 hours/day is ~2.4 kWh; at an assumed Rs 8/kWh
  that is about Rs 20/day (rough, tariff assumed). Idle draw is lower. Cloud-GPU rental for comparison would be a
  per-hour charge; not needed if the card exists.

## 8. What I would do next
Replace the regex filter with a tested classifier; swap synthetic data for a licensed public cut-off dataset with its
licence stated; add a reranker; measure the 32B model on the workstation and update section 4 with real numbers.
