# Transformers Performance Findings — Master Reference

## FEATURE-PR TRACK (mined Jul 25 2026 — second axis alongside the 23 fixes)

**Merge-rate data (Jan 2025–Jul 2026, 5,803 commits):** most external-friendly categories = GGUF adds (73% external, but OVERSUBSCRIBED — tracker #33260 is a newbie magnet, skip unless fresh arch), SDPA adds to individual models (62%), executorch/export (59%), fast image processors (53%, now 95% saturated). Landmark precedent: #43030 ModernBERT attention refactor (+874/−2021, external) → trust-building arc.

**Ranked opportunities (all duplicate-checked Jul 25):**
- **F-1. SDPA for attention-sink models (gpt_oss, mimo_v2, deepseek_v4)** — TOP PICK, unclaimed. `modeling_gpt_oss.py:403` `_supports_sdpa=False`; `modeling_mimo_v2_flash.py:480` comment admits gap ("no sink API atm (same as gpt-oss)"). Flex already handles sinks via LSE renorm (`integrations/flex_attention.py:348-360`). Build: exact sink emulation in SDPA — zero value-row + per-head bias column in mask; mathematically exact. ~80-150 lines; CPU-verifiable equivalence tests. Adjacent (not dup): issue #41465.
- **F-2. Emu3 fast (torchvision) image processor** — unclaimed; only 3 slow-only models remain (emu3, video_llava, vivit). +400-700 lines, templated (Fuyu #41817), equivalence suite auto-runs. "Closed the initiative" resume line. F-3: video_llava + vivit follow-ups (vivit may want BaseVideoProcessor — ask first).
- **F-4. GOT-OCR2 SDPA/flex** — unclaimed; `modular_got_ocr2.py:206-208` disables all backends but SAM ships the exact rel-pos-bias recipe (`modeling_sam.py:1003`). +100-200 lines.
- **F-5. Flex for Swin-family/BEiT rel-pos-bias models** — unclaimed; `flex_attention.py:295-296` supports position_bias already; `modeling_swin.py:722-723` hard-disables. Benchmark first; pitch as uniformity if flat.
- **F-6. Flash-DMATTN kernels backend** — MAINTAINER-INVITED (issue #41465: Arthur "happy to do so via the `kernels` package"; vasqu notes training + T5/MPNet extension). Flagship after credibility; claim in-thread first; coordinate w/ LoserCheems + vasqu. +200-400.
- **F-8. torch.export for Mistral3/Pixtral** — unclaimed; `integrations/executorch.py:34` VLM wrapper only targets SmolVLM2; template = Gemma3 #37728 (+369/9 files). Coordinate w/ Guang Yang (Meta).
- **F-9. Attention-interface refactor for legacy model** — 189 files bypass the interface. T5 CLAIMED (#47014+#42453, do not touch); wav2vec2 wave started (#46196). Best open: MPNet (sentence-transformers traffic, rel-pos flex-able), SpeechT5, LUKE, Tapas. #43030-shaped, large.
- **F-10. RMSNorm kernel adoption sweep** (llama4, pixtral, helium, chameleon, mllama, idefics2/3) — ~20 lines, rider-only (busywork risk). DO NOT touch Gemma-family/qwen3_5 ((1+weight) semantics incompatible).
- **F-11. MegaBlocks MoE adoption beyond gpt_oss** — investigate-first; kernel-side maintenance risk (llama4 entry commented "No longer maintained", hub_kernels.py:118-125); ask Mekkouri/Benayoun.
- **AVOID (gaps-for-a-reason):** CB hybrid/linear-attention + VLM support (core roadmap); Llama4 `_supports_flash_attn=False` (chunked-attn correctness, deliberate); MLA/DSA flex opt-outs (indexer semantics); GGUF popular archs (claimed: GLM-4-MoE #46718, DeepSeek-V3 #47421, Llama4 #45546, OLMo-2/Cohere2/Phi-2 claimed in #33260).

**Strategy:** F-1 (bridge: feature PR with fix-portfolio discipline, same masking/attention neighborhood as Findings 3/20/21, reviewed by vasqu/Cyril) → F-2 (breadth, near-zero rejection risk) → F-6 (flagship, maintainer-invited). Comment on issue before each PR per policy.

## CAPABILITY-FLAG CENSUS (Jul 25 2026 — exhaustive backing behind the FEATURE-PR TRACK; 476 model dirs @ e9c987a3)

**v5 polarity note:** all attention/compile capability flags default to **False** on `PreTrainedModel` (`modeling_utils.py:1240-1252`); models opt IN. Gap census = explicit-False (rare, usually commented = DELIBERATE) + unset (large; splits into interface-ready models missing one flag = small flag+glue PRs, vs legacy pre-interface models = F-9-style refactor first). v5 purged `models/deprecated/` (empty dir) and killed `_supports_cache_class`/`_supports_quantized_cache`/`_supports_static_cache`; FA2/3/4 merged into single `_supports_flash_attn`.

**Flag glossary (name → meaning; default False/None unless noted):** `_supports_sdpa` / `_supports_flash_attn` / `_supports_flex_attn` (attention-interface backends verified); `_compatible_flash_implementations` (model-vouched hub flash kernels; only gpt_oss, mimo_v2_flash, minimax_m3_vl, openai_privacy_filter set it); `_supports_attention_backend` (TGI/vLLM contract: interface fns + kwargs passthrough + tensor logit slicing + TP plan); `_can_compile_fullgraph` (generate auto-compile w/ static cache); `supports_gradient_checkpointing` (374 set True); `_can_checkpoint_with_cache` (per-layer, `modeling_layers.py`); `_is_stateful` (23 SSM/recurrent models); `_can_record_outputs` (capture_outputs map); `_no_split_modules`/`_skip_keys_device_placement` (device_map); `_keep_in_fp32_modules[_strict]`; `_tied_weights_keys` (now a dict, regex-capable); `main_input_name`/`input_modalities`; `_supports_flat_attention_mask` (ASR-only: parakeet/lasr/nemotron); generation-side derived METHODS `_supports_default_dynamic_cache()`/`_supports_logits_to_keep()`. Test-level: `test_torch_exportable` (default True, 32 dirs opt out), `model_split_percents` (103 overrides), `test_all_params_have_gradient` (30 opt-outs, ~all MoE), `is_pipeline_test_to_skip` (59 overrides).

**Per-flag gap tables:**

| Flag | True / False / unset (mixed) | Explicit-False breakdown | Mainstream in unset |
|---|---|---|---|
| `_supports_sdpa` | 313 / 11 / 150 (2) | DELIBERATE: gpt_oss+mimo_v2_flash+openai_privacy_filter (sinks = F-1), deepseek_v4 (DSA), got_ocr2 (F-4), videoprism (attn capping), wavlm (gated rel-pos). ACTIONABLE-investigate: nllb_moe + pegasus_x ("flaky logits"), speech_to_text + time_series_transformer ("tests would need a rewrite") | unset = legacy pre-interface set: t5/mt5/umt5/longt5/switch (CLAIMED #47014 "T5 family" — dead), mpnet/luke/tapas/speecht5 (F-9), deberta(v2), longformer, big_bird, bloom, mpt, gptj, codegen, layoutlm*, seamless_m4t*, wav2vec2_bert/conformer — refactor-sized, not flag flips |
| `_supports_flash_attn` | 289 / 25 / 156 (6) | KERNEL-SIDE: lasr/parakeet/nemotron-ASR ("TODO @eustlb when FA supports custom attention bias"), doge (F-6 dmattn), deepseek_v32+glm_moe_dsa ("flash-mla kernels need a bit more work"), minimax_m3_vl (MiniMaxAI/msa hub kernel). DELIBERATE: llama4 (chunked), swin/beit (rel-pos), siglip2 (naflex nn.MHA pooling), blip_2+granite_speech* (fp32 Q-Former), t5gemma2 (mask merge), blt, sam3_lite_text, wavlm, idefics. ACTIONABLE: kosmos2 ("cuda device errors" = debug), nllb_moe, speech_to_text/time_series_transformer (test rewrite) | unset adds sam/sam_hq (rel-pos = blocked), depth_pro, eomt, timesfm |
| `_supports_flex_attn` | 230 / 19 / 224 (3) | DELIBERATE: beit/swin (F-5), nemotron-ASR family (float rel-pos mask), lasr (padding), siglip2, mimo_v2_flash (asym head dim, non-pow2), MLA/DSA. ACTIONABLE by own comment: sew ("needs a proper look into mask creation") | **GOLDMINE: 61 models with sdpa+flash=True, flex unset** — entire Qwen-VL family (qwen2_vl/qwen2_5_vl/qwen3_vl + omni/audio/asr; text stack = Qwen3 which has flex elsewhere → blocker at most vision-tower varlen), gpt2, gpt_bigcode, falcon, stablelm, persimmon, olmoe, moonshine, glm4v/glm46v, kimi_k25, minicpmv4_6, video_llava, video_llama_3, janus, deepseek_vl*, sam2, vjepa2. Hybrid-SSM (jamba/bamba/zamba/falcon_h1/qwen3_5*/qwen3_next) = roadmap AVOID |
| `_supports_attention_backend` | 252 / 3 / 218 (3) | False: blt, florence2, inkling (rel-pos as eager-only kwarg — DELIBERATE) | interface-ready unset (54): whisper + bart/mbart/marian/pegasus/blenderbot/m2m_100/plbart (enc-dec — contract is decoder-only, UNKNOWN), wav2vec2/hubert/sew/unispeech (plausible post-#46196), colpali/colqwen2/colmodernvbert, stablelm, falcon, dia, musicgen; hybrid-SSM = AVOID |
| `_can_compile_fullgraph` | 195 / 28 / 249 (4) | Commented DELIBERATE: aria/dbrx/jetmoe (MoE dynamic slicing), blt/mllama (heterogeneous static cache), falcon_h1 (KV+SSM), minimax (custom cache), idefics (dynamic control flow). TODO-ACTIONABLE: gpt_neo ("needs a hybrid cache" — gemma2 recipe), longt5 (@raushan owns TODO — coordinate). **Uncommented-False suspicious set (ACTIONABLE-investigate): paligemma (Gemma siblings compile — likely stale), aya_vision, cohere2_vision, qwen2_5_omni, qwen3_omni_moe, lfm2_vl, m2m_100, pix2struct, udop, pop2piano, switch_transformers, zaya, cosmos3_edge, ernie4_5_vl_moe, voxtral_realtime** | gen-capable unset: moshi, musicgen, recurrent_gemma, modernbert_decoder, fuyu, idefics2/3, smolvlm, kosmos2, granite_speech |
| `supports_gradient_checkpointing` | 374 / 15 / 83 (4) | False set is all long-tail (bark, bridgetower, clap, eomt, lightglue, mobilenets, patchts*, superpoint, videomt, timm_backbone) — low PR value | — |
| `test_torch_exportable=False` (tests) | 32 dirs opt out | mainstream: mistral3+pixtral (F-8 — skip), **emu3, hunyuan_vl, kimi_k25, lighton_ocr, phi4_multimodal**, mimi, encodec, esm; rest long-tail | export PRs = 59%-external category |

**Ranked top-20 small feature PRs (excludes curated F-1..F-11; top-10 GitHub-searched Jul 25 2026 — ALL UNCLAIMED; #11-20 not individually searched):**

| # | Target | Why actionable | Fix lives | Est. size |
|---|---|---|---|---|
| 1 | PaliGemma `_can_compile_fullgraph` | uncommented False; Gemma siblings compile; likely stale flag | transformers-only | ~5 + tests |
| 2 | GPT-2 flex | sdpa/flash/backend True, standard attn, no stated blocker | transformers-only | ~10-30 |
| 3 | StableLM + Persimmon flex (batch) | same shape as #2 | transformers-only | ~20 |
| 4 | Falcon flex (+ backend) | interface-ready; masks standardized by #46738 | transformers-only | ~20-40 |
| 5 | GPT-BigCode flex | MQA is flex-compatible | transformers-only | ~20 |
| 6 | Qwen2-VL/2.5-VL/3-VL flex | highest-traffic gap in census; text stack = Qwen3; investigate vision varlen first | transformers-only | ~40-100 |
| 7 | OLMoE flex | MoE affects MLP not attention | transformers-only | ~15 |
| 8 | Moonshine flex | enc-dec ASR, interface-ready, on-device traffic | transformers-only | ~20-40 |
| 9 | `sew` flex | own comment invites it ("needs a proper look") | transformers-only | ~30 |
| 10 | GPT-Neo fullgraph | in-code TODO "needs a hybrid cache"; gemma2 recipe exists | transformers-only | ~40-80 |
| 11 | Pegasus-X SDPA | "Flaky logits" = tolerance investigation; #47014 shows appetite | transformers-only | ~30 + investigation |
| 12 | Kosmos-2 flash | "cuda device errors" = debug, not design | transformers-only | investigation |
| 13 | NLLB-MoE SDPA/flash/flex | comment admits time-budget issue | transformers-only | ~40-80 |
| 14 | glm4v/glm46v/kimi_k25/minicpmv4_6/video_llama_3 flex | rides #6's recipe; batch via modular | transformers-only | small each |
| 15 | Aya Vision + Cohere2 Vision fullgraph | uncommented False, llava-shaped siblings True | transformers-only | ~10 + tests |
| 16 | hunyuan_vl / kimi_k25 torch.export | exportable=False, unclaimed; template = Gemma3 #37728 | transformers-only | ~100-300 |
| 17 | phi4_multimodal / lighton_ocr export | same category, OCR/multimodal traffic | transformers-only | ~100-300 |
| 18 | wav2vec2/hubert `_supports_attention_backend` | capstone of #46196 wave | transformers-only | ~30-60 |
| 19 | colpali/colqwen2 `_supports_attention_backend` | backbones already have it; serving traffic | transformers-only | ~15 each |
| 20 | SAM2/vjepa2 flex (rel-pos score_mod) | F-4/F-5 recipe applied to SAM2 | transformers-only | ~60-120 |

Kernel-side (separate repo, NOT flag+glue): FA custom-bias for parakeet/lasr/nemotron-ASR (flash-attn feature; F-6 adjacency), flash-mla for deepseek_v32/glm_moe_dsa (hub kernels), MiniMaxAI/msa for minimax_m3_vl.

**Census caveats:** flag census is complete/mechanical (regex over all modeling_*.py in 476 dirs; modular-generated files carry flags explicitly; 10+ surprising entries spot-read incl. Qwen-VL flex, whisper backend, t5, mpnet). NOT verified: that each interface-ready flex gap actually *works* — each needs a local flex forward-parity check before PRing. Mixed-flag dirs (esm, evolla, pp_doclayout_v2, blip_2, sam3_lite_text) summarized, not per-class mapped. Duplicate-checks sampled: top-10 candidates only (qwen2.5-vl flex, gpt2 flex, whisper backend, siglip2 flash, pegasus_x sdpa, kosmos-2 flash, paligemma fullgraph, moonshine flex, falcon flex, hunyuan_vl export) — all clean; only claim found = #47014 (T5-family SDPA). Pipeline-skip overrides (59) and model_split_percents (103) counted, not read. Policy reminder: batch tiny flag flips with their parity tests — a lone one-line flip risks busywork rejection.

## NEW-MODEL TRACK (censused Jul 26 2026 — third axis alongside the 23 fixes and the FEATURE-PR TRACK)

**Method:** 686 in-tree model_types (CONFIG_MAPPING_NAMES @ e9c987a3) vs top-15-by-downloads for 45 major orgs (622 repos); 262 candidates' config.json fetched for true model_type+auto_map; ~30 GitHub title-searches for claim status. Downloads = 30-day figures.

**Corrections (in-tree despite custom_code tags / stale premises):** `cohere_asr` (CohereLabs/cohere-transcribe — the "absent" premise was WRONG), `paddleocr_vl`, `deepseek_ocr2`, `jina_embeddings_v3`, `minicpmv4_6`, `got_ocr2`, `internvl` (OpenGVLab originals = unconverted-checkpoint gap only, `*-HF` twins exist), `ernie4_5_vl_moe`, `lfm2` (covers LFM2.5), `kimi_k25`, `laguna`, `nomic_bert`, `aimv2`, `florence2`, `minimax_m3_vl` — all SUPPORTED.

**TIER A — runnable on one A100 + demand + unclaimed (all 10):**

| # | Model (model_type) | dl/mo | Size | Claim status |
|---|---|---|---|---|
| 1 | Alibaba-NLP GTE-v1.5 family (`new`) | 3.6M (5 repos) | 137–434M | UNCLAIMED; issue #35568 closed w/o action |
| 2 | h2oai/h2ovl-mississippi (`h2ovl_chat`) | 2.4M | 0.8–2B | UNCLAIMED — zero GH mentions ever; InternVL-derivative |
| 3 | vikhyatk/moondream2 (`moondream1`) | 2.3M | ~2B | UNCLAIMED (only old closed bug issue); vendor-churn risk |
| 4 | nvidia/llama-nemotron-rerank-1b-v2 (`llama_bidirec`) | 1.1M | 1B | UNCLAIMED — zero issues/PRs |
| 5 | jinaai v5/v4/jina-clip-v2 (`jina_embeddings_v5`, none, `jina_clip`) | 2.7M comb. | small | UNCLAIMED; v3 precedent merged (#44251) |
| 6 | microsoft/Phi-3.5-vision (`phi3_v`) | 1.27M | 4B | PR #36036 closed-unmerged Jan 2026 — free to restart |
| 7 | moonshotai/Kimi-VL-A3B (`kimi_vl`) | 550K | 16B-A3B | UNCLAIMED — zero GH hits |
| 8 | dots-studio/dots.ocr + dots.mocr (`dots_ocr`) | 740K comb. | 3B | UNCLAIMED (GH "." search caveat) |
| 9 | allenai/Molmo2 (`molmo2`) | 510K | 4–8B | 3 dormant Molmo PRs (#33962/#34179/#43448), none covers Molmo2 — COORDINATE first |
| 10 | inclusionAI LLaDA2-mini/flash (`llada2_moe`) | 300K | 16B-A1B | UNCLAIMED; diffusion-LM = nonstandard generate loop |

**TIER B (don't PR):** CLAIMED-active: baidu Unlimited-OCR #46836, Nemotron-Omni #46509, LocateAnything #46300+#46776, Step-3.7-Flash #46562+#46658, MOSS-TTS #46447, MiniCPM-V-4.5/o (stale vendor PRs #40448/#37917 — nudge, don't duplicate), Kimi-Audio #37846, Intern-S1 #39722, Baichuan-M1 #37150. TOO-BIG/political: grok-2 (unloadable at all — no model_type/auto_map; biggest major-lab gap; #29704 dormant), step3 321B, Intern-S1-Pro 241B, Ring-2.5-1T, HunyuanImage-3 80B (borderline diffusers scope), Skywork-R1V 38B, Kimi-Linear 48B (exotic KDA).

**TIER C (legacy/deliberate/niche — one line):** internlm2/Baichuan/chatglm2 (vendors chose remote code), OpenELM (1.6M dl but 2024-era, apple abandoned), FastVLM (PR died), MiniMax-VL-01/M1, Ling/Ring bailing, Ouro, Sa2VA, Nanbeige4.2 + Motif-3 (released <1wk before census), BigVGAN standalone, jina-colbert-v2 (PyLate scope); out-of-scope: TRELLIS/Hunyuan3D/SeedVR2/BAGEL/LatentSync/TripoSR/stable-audio-3/PP-OCR det-rec/.nemo/granite-tsfm/VoxCPM2/open_clip repos.

**Top-3 recommendations:** (1) **GTE-v1.5** — best demand-to-effort in census: 3.6M dl, ≤434M params (laptop-validatable), minor BERT variant (RoPE+gated FFN+unpadding), documented demand, never PRed. (2) **llama-nemotron-rerank-1b** — 1.1M dl, 1B, bidirectional-Llama + seq-cls head = ideal `modular_*.py` child of Llama, zero prior threads. (3) **Kimi-VL / dots.ocr** — highest-demand unclaimed VLMs fitting one A100; Kimi-VL rides in-tree kimi_k25 pieces (MoonViT/DS-MoE), dots.ocr = compact Qwen2-VL-style 3B (h2ovl even easier if raw dl is the metric).

**Census caveats:** top-15 per org only (long-tail missed); GH title-search returned 0 for "dots.ocr"/"Kimi-VL"/"Ling" — punctuation/short-token blind spots may hide threads, RE-VERIFY before claiming any entry; gated repos classified via API config field; xai-org exposed only 2 repos.

## PR presentation playbook (from reaction-mining of celebrated external PRs, Jul 25 2026)

- Individual-contributor reaction ceiling ≈ 20-40; that tier = (a) tiny surgical fixes w/ repro (karpathy #17877: +1/−1, 38 reactions, "I randomly noticed a minor bug" + repro + scope statement), (b) precision/perf fixes w/ numerical evidence (danielhanchen #29285: +23/−8, 27 all-hearts, why-explanation + precision images). Mega-reaction PRs are vendor model launches — not our class.
- Winning description structure: plain problem statement → minimal repro / before-after numbers or charts → explicit blast-radius ("X affected, Y not") → tag right maintainer (@ArthurZucker near-universal). No checklists/marketing/walls of text.
- Size is bimodal: model adds (1000s of lines) or fixes (<~40 lines). Mid-size rarely celebrated → keep fix PRs minimal, split big work.
- Repo actively pushes back on agent-written PRs from unknowns (policy boilerplate now embedded in PR templates) → submissions must be human-owned, evidence-driven, AI assistance disclosed.
- Our mapping: Finding 1 = karpathy pattern; Findings 2/3 = danielhanchen pattern (all-hearts genre); Finding 4 = the #30536-sequel chart PR.

> Personal working document. NOT for committing to the repo (excluded via `.git/info/exclude`).
> All line numbers verified against local checkout of `main` at commit `e9c987a3` (synced Jul 23, 2026).
> Purpose: contribution campaign toward HF Transformers core, modeled on Cyril Vallez's path.
>
> **Numbering note:** findings were renumbered Jul 25, 2026 from the old letter codes into a single
> ranked list (1..23). Each heading keeps its old code once, e.g. "Finding 3 (was C1)". Old-code
> provenance: C/G/T/L/V/F = wave 1 (core generate/trainer audit), X/M/P = wave 2 (preprocessing,
> multimodal, TP/correctness), W = wave 3 (PyTorch anti-pattern sweep), S = wave 4 (Cyril-pattern hunt).

---

## 0. Context & Strategy

**Goal:** land incremental, evidence-backed performance PRs on default paths of transformers; build reviewer trust; target HF employment.

**Reference PRs (Cyril Vallez) and their sizes:**

| PR | What | Size |
|---|---|---|
| #30536 | Halved `generate()` memory (KV cache duplication via `torch.cat` + stale refs) | +327/−206, 19 files |
| #31292 | Removed useless full-vocab logits in prefill | +428/−41, 23 files |
| #33570 | Removed CUDA sync in flash-attention decode | **+2/−1, 1 file** — legendary because of the profiler evidence, not the LOC |
| #47170 | (ShareLer, merged by vasqu, Jul 2026) Hoisted FA max-seqlen compute to once-per-forward for vision models | +399/−136, 41 files — precedent for "hoist per-layer FA metadata" |

**Repo policy constraints (from AGENTS.md/CLAUDE.md):**
- Coordinate on existing issues before PRing; never PR someone else's issue without approval.
- Duplicate-check via `gh pr list --search` before proposing.
- No low-value busywork PRs. Disclose AI assistance. First-time contributors shouldn't submit pure-agent PRs — user must own/understand every fix.

**PR sequencing plan (agreed):**
1. Open with Finding 5 (Trainer NaN-filter sync) — tiny, unrejectable, most mainstream audience.
2. Then Finding 3 (sliding-window mask) — the headline chart.
3. Then Finding 4 (DynamicCache preallocation) — the #30536-sequel, proposed after credibility established.
4. Stash items (FA syncs) and mid-tier findings interleaved as opportunities.

**Benchmark methodology (agreed):**
- Library-agnostic scripts: monkeypatch functions at runtime, A/B/A design (stock → patched → restored-stock) in one process. Add a call-counter assertion to prove the patched code is actually hit.
- Correctness gates: fixed-seed generation must produce identical token IDs; forward logits must be bitwise equal (fixes change *when* things are computed, never *what*).
- Mechanism proof: `torch.profiler` counting `cudaStreamSynchronize` events per step/token, before vs after.
- Hardware: FA2 findings need Ampere+ (Colab L4/A100; free T4 CANNOT run FA2). Sync findings need any CUDA GPU (CPU torch is synchronous — syncs don't exist there). Memory findings often CPU-verifiable.

---

## MASTER RANKING (index)

| # | Was | One-liner | Strategic note |
|---|---|---|---|
| 1 | X1 | Assisted/speculative decoding crashes for ALL sliding-window models once context ≥ window | Crash in Cyril's 2-day-old flagship feature; repro + validated fix; act NOW |
| 2 | M1 | Qwen2-VL/2.5-VL video M-RoPE position ids silently WRONG (3 divergences) | Silent wrong output on flagship video VLM; executable reference-diff |
| 3 | C1 | Sliding-window decode all-True mask kills SDPA GQA path → repeat_kv copy/token | 25-30%/2-3× decode tax; new; small fix |
| 4 | G1 | DynamicCache O(T²) re-concatenation on default `generate()` | #30536-sequel |
| 5 | T1 | Default Trainer full GPU sync every micro-batch (`logging_nan_inf_filter`) | 10-30% small-model training; ~6 lines; FIRST PR |
| 6 | S1 | `StaticSlidingWindowLayer` O(W) roll + double-copy per decoded token | 4-15% decode bandwidth, Gemma default path, Cyril's own file |
| 7 | P1 | `BatchEncoding.convert_to_tensors` flatten + `torch.tensor(nested_list)` | 6.4× default collate; CPU-provable; maintainer buy-in exists |
| 8 | P3 (+P4 as 8b) | Video decoding fallback crashes on torchvision ≥0.26; whole-video materialization; sequential decode 5× | Crash + 50 GB transient |
| 9 | X2+X3 | TP save path silently corrupts checkpoints (digit-wildcard mangling + bias→`_bias`) | Silent corruption |
| 10 | L1 | Chunked cross-entropy (deepening of known fp32 upcast) | 30-60% peak mem; coordinate, known territory |
| 11 | M2 | Qwen-VL vision towers: ~100+ serialized tiny SDPA calls + per-layer `.tolist()` sync | Staged fix |
| 12 | W1+W2 | Whisper timestamps: ~4·B syncs/token + pure-Python DTW | Coherent single-file mini-campaign |
| 13 | P2 | Processor placeholder-token counting via `list(tensor).count(int)` | 21-24% of processor call |
| 14 | V1 | Eval loop O(N²) logits re-concat + bf16→fp32 inflation | Mechanics unreported |
| 15 | G2 | TopP full-sorts vocab per token although TopK(50) just ran | Min-p precedent #42248 |
| 16 | X4 | `EmbeddingParallel` all-zero embeddings when vocab % tp_size ≠ 0 | Silent, niche |
| 17 | X5+X6 | longrope inv_freq dead cache guard + yarn `truncate` ignored in nested configs | Tiny, same file |
| 18 | F1 | FA per-layer D2H sync in `_is_packed_sequence` (B=1 decode) | Coordinate on #39213/#47342 |
| 19 | F2 | FA `_get_unpad_data` recomputed per layer with 2 syncs | Bigger design PR after 18 |
| 20 | C2 | Padded-batch decode full repeat_kv materialization | Upstream-blocked; stash |
| 21 | C3 | Per-token GPU syncs in generate/masking | Revive #43085/#43088; stash |
| 22 | G3 | Continuous-batching streaming O(N²) both sides | Stash |
| 23 | G4 | Beam search full-cache `index_select` per step | Minority path; stash |
| 24 | W5 | Gemma-2/VaultGemma/T5Gemma silently drop attn softcapping on DEFAULT sdpa backend — regression of #32865, deleted by #35235 | Silent wrong outputs on flagship Google default path for 19 months; git-archeology proof; highest review-impact class |

---

# FINDINGS (ranked)

### Finding 1 (was X1) — Assisted/speculative decoding CRASHES for ALL sliding-window models once context ≥ window — regression in Cyril's PR #47447, merged Jul 22 2026 (2 days before discovery)

**THE golden opportunity: fresh regression in the reference maintainer's own headline feature, pre-release window, repro + validated fix in hand.**

- **Root cause:** #47447 added `record_past`/`activate_past_recording` to `DynamicSlidingWindowLayer` so assisted decoding works with sliding caches. With `record_past=True` (set unconditionally at `generation/utils.py:3615` and `:2011`), `update` keeps ALL tokens (`cache_utils.py:243-245`) until `crop` — but `get_mask_sizes` (`cache_utils.py:250-260`) still assumes exactly `W-1` tokens once `cumulative_length >= W` → mask built with `kv_length = W-1+q` while keys have length `cached+q` → SDPA shape mismatch (`integrations/sdpa_attention.py:154`).
- **Two repro'd failure paths:** (1) prompt ≥ W (prefill records full prompt, no crop before first verify) — `RuntimeError: size of tensor a (29) must match b (17)`; (2) all candidates accepted → `crop` skipped when `number_of_tokens_to_crop == 0` (`generation/utils.py:3784-3786`) → breaks next round. Path 2 is the COMMON case (well-matched assistant, prompt-lookup decoding).
- **Who pays:** Mistral (W=4096), Gemma-2/3/4, GPT-OSS (W=128 — crashes within ~128 tokens), Llama4 chunked + `assistant_model=` or `prompt_lookup_num_tokens=`.
- **Why tests miss it:** tiny prompts below window; random assistants get rejected each round so crop restores the invariant.
- **Validated fix (~8 lines):** derive from reality in `get_mask_sizes`: `cached = self.keys.shape[-2]; kv_offset = self.cumulative_length - cached; kv_length = cached + query_length` — provably identical to current values in non-recording path; sliding mask function correctly masks extra recorded tokens. Validated: both crashes pass, token-identical to plain greedy.
- **Duplicates:** none. Adjacent: open #47433 (crop(0) semantics — different bug, owned by someone else, don't touch).
- **Verify:** fully CPU. **Action: new issue + PR NOW (pre-release window), reference #47447.**

### Finding 2 (was M1) — Qwen2-VL / Qwen2.5-VL video M-RoPE position ids silently WRONG — 3 divergences from Qwen reference, on default settings

**Flagship deployed video VLM producing off-distribution positions. Executable repro vs verbatim v4 reference algorithm.**

- **Divergence 1 — temporal interval truncated to int** (`modeling_qwen2_5_vl.py:1058`): `time_interval = tokens_per_second * int(next(second_per_grid_ts))`. Reference: per-element `floor(k · spg · tps)` (float mult then `.long()`). Processor sets `spg = temporal_patch_size / sampled_fps` (`processing_qwen2_5_vl.py:146-148`) — routinely fractional. Any sampled fps > 2 → `int(spg)=0` → **all frames get identical temporal position; frame order erased from RoPE**. Lineage: side effect of device-fix #37597 (Apr 2025), carried by v5 rewrite #43972; maintainer PR #45400 *observed* the rounding but didn't fix it.
- **Divergence 2 — text after video resumes at spatial-only offset** (`modeling_qwen2_5_vl.py:1065`): `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`; reference uses global max incl. temporal axis. Diverges at default fps=2 for videos ≳5-10 s. Same pattern in `qwen2_vl/modeling_qwen2_vl.py:1023`; 14 model files share it (needs per-reference audit).
- **Divergence 3 — prompt/decode discontinuity (internally provable):** `mrope_position_deltas` computed from global max (`modeling_qwen2_5_vl.py:1071`), inconsistent with (2) → first generated token jumps (repro: prompt ends at 28, generation starts at 65).
- **Repro numbers:** 30 s video fps=2: trailing text at 22-28 (main) vs 65-71 (reference). fps=2.4: all temporal ids identical (interval=0). Image-only control: bitwise identical (methodology sound).
- **Who pays:** every Qwen2/2.5-VL VIDEO inference/fine-tune, ≥v5.3.0 (div 2,3) / ≥~v4.52 (div 1). Silent degradation of temporal grounding.
- **Fix:** ~10-15 lines per model in modular files (per-element temporal ids; advance by max(temporal, spatial) extent — also fixes the delta gap). Needs tests pinned against official Qwen impl. Changes video outputs intentionally — bring the repro table.
- **Duplicates:** none. Adjacent-fixed: #45325/#45330 (image scaling), #45381/#45400 (temporal offset ordering), #41152/#41153. **Verify: CPU-sufficient.** Optional GPU demo: temporal-ordering VQA before/after.

### Finding 3 (was C1) — Sliding-window decode materializes an all-True mask → disables SDPA GQA fast path → full `repeat_kv` copy of KV window per layer per token

**Status: STRONGEST WAVE-1 FINDING. No existing issue/PR. Verified by executing the masking code (CPU).**

- **Mechanism:** Once context > `sliding_window`, `DynamicSlidingWindowLayer` already crops the cache to the last W−1 tokens (`cache_utils.py:239-241`, kv_offset at `:250-260`), so for a 1-token decode query *every* cached key is inside the window → the materialized `(B,1,1,W)` mask is provably all-True. But `_ignore_causal_mask_sdpa` refuses the skip purely because the window is full: `masking_utils.py:262-263` (`if local_attention_size is not None and kv_length >= local_attention_size: return False`). The non-None mask then fails `use_gqa_in_sdpa` (`integrations/sdpa_attention.py:38` requires `attention_mask is None`) → `repeat_kv` at `sdpa_attention.py:97-102` does `expand().reshape()` = full materialized copy of the KV window, per layer, per token; SDPA also downgrades flash→mem-efficient/math kernel because a mask is passed.
- **Call chain:** `MistralModel.forward` → `create_sliding_window_causal_mask` (`modeling_mistral.py:372`; Gemma3 `modeling_gemma3.py:565-566`) → `sdpa_mask` with `local_size` (`masking_utils.py:1212-1226`) → `_ignore_causal_mask_sdpa` refusal → mask reaches `sdpa_attention_forward`.
- **Executed verification:** decode-step mask with Mistral config + DynamicCache: shape (1,1,1,16), allTrue=True (both with generate's all-ones 2D mask and with mask=None). Multi-token continuation (q=3): allTrue=False → q>1 case genuinely needs the mask, must stay excluded from the skip.
- **Quantified (Mistral-7B: 32 sliding layers, W=4096, n_rep=4, D=128, bf16):** KV window/layer 16.8 MB; repeat_kv reads 16.8 + writes 67.1; SDPA reads 67.1 instead of 16.8 → +134 MB/layer → **+4.3 GB traffic per decoded token**; vs ~15 GB baseline → **~+25-30% decode latency at batch 1; ~2-3× at batch 8**. Gemma-2-9B: ~+2.1 GB/step (~+10% @B1). Gemma-3-12B (W=1024, 5/6 layers sliding — exceeded in nearly every real chat): ~+1 GB/step (~4% @B1, ~30% @B8).
- **Fix (~20-40 lines, `masking_utils.py` + small cache flag):** allow is_causal-style skip when q_length==1 AND cache layer is dynamic sliding (exact cropping guaranteed; static sliding caches must keep mask) AND trailing-window slice of padding mask is all ones. Then mask=None → `enable_gqa` re-engages. Risk: low-moderate.
- **Duplicates:** none (`sliding window mask sdpa`, `repeat_kv slow`, `use_gqa_in_sdpa` searched). Condition dates to mask rewrite #37866. Adjacent-not-duplicate: #46168 (eager SWA prefill), #45776 (enable_gqa gating), #47094 (StaticCache prefill FA slicing).
- **Benchmark:** GPU required. Mistral-7B or Gemma-2-9B, prompt > window (6K), generate 256, B=1 and B=8, before/after; profiler shows the 2 repeat_kv copy kernels per layer disappearing. Mask-is-None behavior change itself is CPU-verifiable.

### Finding 4 (was G1) — DynamicCache re-concatenates the entire KV cache every token — O(T²) traffic on default `generate()`

**Status: #30536-class. No duplicate found.**

- **Mechanism:** default cache is DynamicCache (`generation/utils.py:1995-1997`); every layer every step: `self.keys = torch.cat([self.keys, key_states], dim=-2)` (`cache_utils.py:143-145`) → new allocation + full copy of existing cache. Auto-compile excludes DynamicCache (`utils.py:2120`), so eager always.
- **Quantified (Llama-3.1-8B, bf16: 128 KB KV/token):** per step at context S ≈ 2·S·128 KB traffic. S=8K → 2.1 GB/step ≈ +16% of ~8 ms A100 step; S=32K → 8.4 GB/step ≈ +40-60%. Total for P=T=2048: ~1.6 TB pure copy traffic. Plus 64 cat kernels + 64 alloc/frees per step; monotonically growing sizes → allocator fragmentation, reserved memory balloons ~1.5-2× live KV. **CPU-measured: 17.3 ms/step (cat) vs 0.128 ms/step (preallocated index_copy_) = 135× on the cache op** (script was at /tmp/gen_audit_bench.py — regenerate if needed).
- **Cross-check (core audit's independent estimate):** per-step overhead ~12% of step traffic at T=8K, ~5% at 4K — the honest range is 5-40% depending on context length; agreement that it dominates at ≥8K.
- **Comparison:** vLLM paged blocks, llama.cpp preallocation, gpt-fast/HF StaticCache — all preallocate. Per-token cat growth is a documented PyTorch anti-pattern.
- **Fix (~50-80 lines, `cache_utils.py`):** amortized geometric growth in DynamicLayer — preallocate capacity (generate() knows max_length and could hint), write via slice/index_copy_, return `self.keys[:, :, :length]` views, double when full. **Risks:** returned views non-contiguous in dim -2 (SDPA/FA2/eager accept strided KV; anything calling `.contiguous()` copies — no worse than today); `crop`/`batch_select_indices`/beam ops need adjustment.
- **Duplicates:** none (`DynamicCache preallocate/quadratic/cat slow` searched). #40328 (compile+DynamicCache) orthogonal; #43089 (syncs) different; #30862 inverse.
- **Benchmark:** GPU for end-to-end: 8B, prompt 1K, generate 1K/8K/32K, ms/token curve + max_memory_reserved + nsys of one step (64 cat kernels visible). CPU microbench isolates cache-op cost (done, 135×).

### Finding 5 (was T1) — Default Trainer forces full GPU sync every micro-batch (`logging_nan_inf_filter`) + blocking H2D copies

**Status: UNREPORTED. Direct #33570 analogue on the most mainstream path. Recommended FIRST PR.**

- **Mechanism A:** `trainer.py:1760-1763`: `if self.args.logging_nan_inf_filter and not is_torch_xla_available() and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))` — `tr_loss_step` is a 0-dim CUDA tensor (`training_step` returns `loss.detach()`, trainer.py:1962); Python `or` → `Tensor.__bool__` → `.item()` → `cudaStreamSynchronize` waiting for full fwd+bwd. Default ON (`training_args.py:979-983`). Irony: comment at trainer.py:1515-1516 says `_tr_loss` stays on-device to "avoid costly `.item()` syncs".
- **Mechanism B:** Trainer builds accelerate `DataLoaderConfiguration` with `non_blocking=False` default (`trainer_pt_utils.py:1203-1209`) while `dataloader_pin_memory=True` — synchronous cudaMemcpy per batch forfeits pinning. Both must be fixed together.
- **Quantified:** ≥2 full-stream serializations per micro-batch = zero CPU run-ahead; 2-10 ms forced GPU idle/step. 0.5-2B model, seq 512-1024 (common SFT regime, 20-80 ms steps): **10-30% wall-clock**. 8B, 8×4096 on H100: ~0.5-2%.
- **Fix (~6 lines):** on-device filter — `is_finite = torch.isfinite(tr_loss_step); tr_loss += torch.where(is_finite, tr_loss_step, tr_loss/(1+steps_since_log))`; flip `non_blocking=True` when pin_memory on (accelerate ≥0.30). Semantics preserved exactly, evaluated lazily.
- **Duplicates:** none (`logging_nan_inf_filter`, `isnan isinf gpu sync`, `non_blocking pin_memory trainer` searched).
- **Benchmark:** GPU: 0.5B model, seq 512, torch.profiler before/after — cudaStreamSynchronize disappears + step-time delta. ~20-line script.

### Finding 6 (was S1) — `StaticSlidingWindowLayer.update`: O(W) roll + double-copy per decoded token — default Gemma-2/3 cache path

**Bulletproof, novel, in Cyril's home file (`cache_utils.py`). 4-15% of decode bandwidth.**

- **Mechanism (`cache_utils.py:522-540`):** once window full, appending ONE token does `self.keys.roll(-1, dims=-2)` (read W + write W, full rotated copy) + tensor-index write + `self.keys.copy_(new_keys)` (read W + write W again, to keep static dynamo address) = **4 full-window traversals × K,V × per sliding layer × per token**, where a ring buffer needs a single 1-token write. Structure says O(1); torch ops do O(W). Comment cites pytorch#159855 (dynamo cat bug) as why not cat — but `index_copy_` (which `StaticLayer.update` already uses under compile) is unaffected.
- **Default-path chain (verified):** Gemma-2/3 hub `generation_config.json` ships `cache_implementation: "hybrid"` → mapped to `StaticCache` → `StaticSlidingWindowLayer` per sliding layer → roll branch every decode step past W. Cache persists across generate calls (`generation/utils.py:1820-1848`) — steady state.
- **Arithmetic (bf16, B=1):** Gemma-3-12B (28 sliding layers, W=1024, D=256): ~**0.9 GB overhead traffic/token** ≈ 4% of decode bandwidth. Gemma-2-9B (21 layers, W=4096): ~**2.8 GB/token ≈ 13-15%**. Mistral-style W=4096 static ≈ 16%. **CPU-verified on the real class: roll 290 µs vs ring write 1.6 µs per layer-step (~180×).** Honest caveat: compiled mode may fuse roll+copy (gain bounded below by ~2 traversals) — state up front.
- **Sibling:** `DynamicSlidingWindowLayer.update` cat+slice per token once full = same waste class on the dynamic path (distinct from Finding 4's O(T²)-over-full-length). Ring redesign covers both.
- **Fix + correctness bar (RECON COMPLETE):** recommended design = **ring-for-decode + linearize/cat-on-q>1**: not-full path unchanged; full+q=1 (hot) `index_copy_` at `(cum % W)` = O(1) vs today's ~4 full-window traversals; full+q>1 (cold, ~once per chat turn) linearize ring + existing cat fallback. Ring write idiom (`index_copy_` with tensor index) is exactly what `StaticLayer.update` already uses → compile/cudagraph-safe, same 2-3-graph regime Cyril endorsed.
- **#44609 recon (corrected framing):** issue was about torch.compile RECOMPILATION, not bandwidth — nobody has ever raised the bandwidth cost (unclaimed argument). Author jazpurTT's hack was an ALWAYS-ROLL branchless variant (not a ring; still O(W)) + constant `get_mask_sizes`. Cyril's exact objection (2026-03-12): sliding layer "*needs* to be dynamic by essence, to be logically correct" and the hack is "logically incorrect in general, as in some cases you will not return enough past states" — i.e., full-cache q>1 must return W−1+q states (cache_utils.py:541-544, 572-576 "we should return the whole states... otherwise we lose some context"; `get_mask_sizes` returns dynamic `W+q-1` at :585-586).
- **Correctness traps for the ring (verified):** (1) q=1 full-window attention is permutation-invariant over KV slots (RoPE applied BEFORE caching — gemma2 modeling :275 vs :278) → pattern mask fine; BUT (2) `prepare_padding_mask` maps padding bits by linear slot rule (masking_utils.py:205-214) → left-padded batched decode breaks unless rotation-mapped; (3) index-based `or_mask_function` overlays — Gemma3 itself passes `_bidirectional_window_overlay` (modeling_gemma3.py:559-566) — assume kv_idx = absolute position → rotation must be exposed generically; (4) mask tensor is already materialized every compiled decode step (masking_utils.py:1184, 262-263) so rotation-aware mask adds NO new per-step cost; (5) StaticSlidingWindowLayer has no crop/record_past → assisted decoding can't hit it; q>1 fires only on prefill continuation/prefix reuse = cold. Position-indexed-writes-without-mask-awareness design is DOA (keys stored post-RoPE).
- **No existing PR (verified exhaustively):** closest are #37370 (introduced the roll, Apr 2025), #39797 (carried it), #45745 (open+stale, static crop, orthogonal), #45940 (closed 1-line mask fix, rejected), #47447 (Cyril 2 days ago, DYNAMIC layer rollback only). No evidence Cyril plans the bandwidth fix — his in-code TODO points the OPPOSITE way ("we should use a much simpler `cat`", blocked by pytorch#159855) = he weights simplicity over update bandwidth → **the proposal must win on measured numbers, incl. compiled mode** (Inductor may fuse roll+copy; benchmark eager AND compiled).
- **Approval path:** NEW issue (not #44609 — that was compile-topic, closed) with: bandwidth arithmetic + eager/compiled benchmarks + exact-match logits test matrix (q=1 decode, q>1 continuation, left-padded batch, Gemma3 bidirectional overlay) + design doc showing the q>1 contract preserved by construction. Risks: collision (Cyril active in this exact area right now), his simplicity preference, review burden on mask plumbing.

### Finding 7 (was P1) — `BatchEncoding.convert_to_tensors`: full `flatten()` pass + `torch.tensor(nested_list)` — 96% of default Trainer collate cost; 6.4× collate speedup measured

- **Two wastes:** (1) `tokenization_utils_base.py:702` — `len(flatten(value)) == 0` materializes a flat copy of the whole batch per key just to test emptiness (3.47 ms/key @32×2340); (2) `:704` — `torch.tensor(nested_list)` is 4.3× slower than `torch.from_numpy(np.asarray(value))` (3.87 vs 0.86 ms); list-of-np special case at `:700-701` never fires (Rust tokenizer returns plain lists).
- **Call chains:** every `tokenizer(..., return_tensors="pt")` (`tokenization_utils_tokenizers.py:996`); **every default-Trainer step** via `DataCollatorWithPadding` (`trainer.py:544` → `tokenization_utils_base.py:2769`); `apply_chat_template` (`:3136-3144`).
- **Measured:** Trainer collate B=64 L~1024: 12.94 ms stock, conversion = 96%; patched = 2.02 ms (**6.4×, bit-identical asserted**). `__call__` B=32 pt: conversion 36% of call. With `dataloader_num_workers=0` (default) this is synchronous main-process time between steps.
- **Fix (~10-15 lines, `tokenization_utils_base.py`):** O(depth) emptiness peek + np-bridge with `torch.tensor` fallback for ragged. In-repo precedent: `BatchFeature._get_is_as_tensor_fns` (`feature_extraction_utils.py:130-142`) already np-bridges. Watch offset_mapping tuples + empty-input dtype.
- **Duplicates:** #33855 (Oct 2024, BatchFeature-theme, LysandreJik: "feel free to open a PR", stale-closed, never delivered) — prior maintainer buy-in to cite. **Verify: CPU-representative (done).**

### Finding 8 (was P3; 8b was P4) — Default video-decoding fallback CRASHES on torchvision ≥0.26; materializes ENTIRE video on 0.22-0.25; sequential frame decode 5× slower than seek

**8a (was P3) — crash + whole-video materialization:**

- `BaseVideoProcessor.fetch_videos` (`video_processing_utils.py:827-838`) hardcodes torchcodec-else-torchvision; pyav/decord/opencv never tried (though pyav is `load_video`'s documented default, `video_utils.py:635`). torchvision ≥0.26 removed `io.read_video` → `AttributeError` reproduced on 0.28; torchcodec is NOT a transformers dependency → stock mid-2026 install crashes on `processor(videos=...)` / chat template with video (funnel: `processing_utils.py:2135-2141` → `video_processing_utils.py:294`).
- On torchvision 0.22-0.25: `read_video(start_pts=0, end_pts=None)` decodes+materializes the WHOLE video TCHW before sampling (`video_utils.py:538-556`): 10-min 720p ≈ ~50 GB transient for 16 frames.
- **Fix (~20-40 lines):** backend fallback chain (torchcodec → decord → pyav → opencv) + version-gate the torchvision reader. **Duplicates:** none. **Verify:** 3-line CPU repro.

**8b (was P4) — `read_video_pyav`/`read_video_opencv` decode every frame sequentially — seek-based is 5× faster, bit-identical:**

- `video_utils.py:497-505` (pyav: decodes from frame 0, filters `i in indices` — O(S) ndarray scan per frame too); `:384-395` (opencv: `read()` full decode per frame instead of `grab()`). Default of public `load_video` (backend="pyav") and the tier promoted by the 8a fix.
- **Measured:** 3-min 720p GOP=60, 16 frames: 3.00 s → 0.59 s seek-based (5.0×), `np.array_equal`. Caveats: dense sampling / large GOP needs density heuristic (`len(indices)*est_gop > total_frames → sequential`); B-frame pts matching required. ~40-60 lines. **Duplicates:** none. Bundle 8b with 8a. **Verify: CPU (script synthesizes own fixtures).**

### Finding 9 (was X2+X3) — Tensor-parallel save path silently corrupts checkpoints (two coupled bugs)

**9a (was X2) — TP save path: `re.sub(r"\d+","*")` mangles `fc1`→`fc*` — local shards silently saved as full weights:**

- Load path uses `replace_layer_number_by_wildcard` (dot-delimited only, `integrations/tensor_parallel.py:125-132`); save path `gather_state_dict_for_save` (`:1462-1465`) wildcards ALL digits → plan lookup misses → "not sharded, keep as-is" (`:1479-1481`) → checkpoint has 1/tp_size rows under full key. Verified with real PhiForCausalLM + 1-proc gloo: load says colwise/rowwise for `mlp.fc1/fc2.{weight,bias}`, save sees None for all 8 params.
- Affects: Phi, nanochat, cosmos3_edge, ernie4_5_vl_moe, Llama4 vision adapter. Silent at save; reload shape-mismatch. **Fix ~5 lines** (reuse `_get_parameter_tp_plan`). No duplicates.

**9b (was X3) — `RowwiseParallel` forward pre-hook permanently renames `bias`→`_bias` in state_dict:**

- `_prepare_input_fn` (`tensor_parallel.py:886-888`): `mod._bias = mod.bias; mod.bias = None` — never restored; `nn.Module.__setattr__` re-registers. After ONE forward, `fc2.bias`/`dense.bias` vanish from `state_dict()`, `_bias` appears; reload silently re-inits real biases; combined with 9a-style gather, `_bias` is gathered replicated → tp_size× garbage tensor. Verified.
- Affects: TP training of biased rowwise models — Phi, GPT-OSS (attention_bias=True), Falcon, StarCoder2. **Fix ~6 lines** (restore in `_prepare_output_fn` or stop mutating params in hooks). No duplicates.

### Finding 10 (was L1) — Training loss: chunked cross-entropy (deepening of KNOWN fp32 upcast)

**Status: known territory (Liger/torchtune); frame as "bring best practice into core"; coordinate first (Liger-integration discussion, ~#34191 era).**

- **Verified anatomy (saved_tensors_hooks, CPU torch 2.13):** with `.float()` (`loss_utils.py:59`, reached from `modeling_llama.py:491` via LOSS_MAPPING), autograd retains exactly ONE full fp32 tensor into backward (log_softmax output; saved twice but same storage via data_ptr). The fp32 upcast copy is transient (dies after CE forward). bf16 logits NOT retained (linear backward saves input+weight). "Double materialization by autograd" hypothesis = FALSE. Label shift copies are noise (~65 KB).
- **Peak arithmetic (Llama-3-8B, V=128256, 8192-token micro-batch = 1.05e9 elements):** CE forward peak 10 B/elt = **10.5 GB** (bf16 2.1 + fp32 copy 4.2 + fp32 log-softmax 4.2); retained into bwd 4.2 GB; bwd transient ~12.6 GB. Without upcast: 4.2/2.1/6.3 GB. Chunked (Liger/torchtune style): ~0.2/~0/~0.5 GB.
- **Torch reference fact:** log_softmax/CE kernels already accumulate in fp32 for half/bf16 inputs (AccumulateType, PyTorch PR #24457; `CUDA_ACC_TYPE(BFloat16, float)`). Upcast buys fp32 *storage* of log-probs. Dropping it: loss off 2.4e-3 relative, grad max rel error ~5e-2 → HF's upcast is defensible; honest fix is chunked CE (upcast 1-4K rows at a time), fp32-equivalent grads.
- **Fix:** ~40 lines in `fixed_cross_entropy` (`loss_utils.py:49-70` area). **Gain: 30-60% of peak memory in memory-constrained fine-tunes** (loss spike IS the peak for 8B QLoRA/short-context). CPU-verifiable via allocator stats on scaled vocab.

### Finding 11 (was M2) — Qwen-VL-family vision towers: ~100+ serialized tiny SDPA calls per windowed layer + `.tolist()` device sync per layer — default SDPA path, 19 model files

- `Qwen2_5_VLVisionAttention.forward` non-flash branch (`modeling_qwen2_5_vl.py:269-290`; same in qwen3_vl `:249-267`, glm4v, glm_ocr, kimi_k25, ernie4_5_vl_moe, video_llama_3, paddleocr_vl, hunyuan_vl, minicpmv4_6, qwen2_5_omni, qwen3_omni_moe, cosmos3_edge, exaone4_5, …): `lengths.tolist()` (sync × 32 layers) + per-window `torch.split` + sequential SDPA per ≤64-patch window. 1288×952 image → 108 windows × 28 windowed layers ≈ ~3,000 serialized SDPA dispatches + 32 syncs per image forward. Qwen3-VL chunks per-frame/per-image (serial over frames).
- **Default confirmed:** attn resolves None→"sdpa" (`modeling_utils.py:1940`); flash branch only if explicitly requested (`:251`).
- **History for the narrative:** loop introduced by #37363 (Jul 2025) replacing single-SDPA-with-block-diag-mask — was itself a win (A100: 4.30→1.88 s/tower forward) but its own data shows flash at 0.22 s = **8.5× residual gap**, mostly serialization. Do NOT revert to masks over packed sequence.
- **Measured here:** MPS fp16 real geometry: 13.20 ms/layer chunked vs 8.58 batched-padded-windows (1.5×; tower −22%), max|Δ|=0.00e+00 (bool key-padding mask). CPU fp32: chunked is FASTEST (dispatch-bound pathology) → **fix must be device-gated**.
- **Staged fix:** (1) mini-PR ~10-20 lines: hoist `cu_seqlens.diff().tolist()` + split boundaries out of the 32 layers (loop-invariant) — kills 32 syncs, zero numerics risk, all 19 files via modular; (2) main PR ~60-100 lines: per-forward gather index + bool mask → one SDPA over (num_windows, heads, 64, head_dim); est. 2.5-4× tower speedup on CUDA (medium-high confidence, needs GPU run). Keep chunk loop for heterogeneous full-attn chunks.
- **Duplicates:** none. Cite #37363 (baseline numbers), #47170 (hoisting precedent), #45816. **GPU required for headline; MPS/CPU evidence in hand.**

### Finding 12 (was W1+W2) — Whisper timestamps: ~4·B device syncs per decoded token + pure-Python O(M×N) DTW

**Strategy: 12a+12b = coherent "make Whisper timestamps fast" mini-campaign — same file ownership, same reviewers, CPU-provable correctness for both.**

**12a (was W1) — `WhisperTimeStampLogitsProcessor`: ~4·B device syncs PER DECODED TOKEN on default long-form transcription:**

- **Syncs per token (all in `__call__`, `logits_process.py`):** `.tolist()` per batch row (`:2013`); boolean-mask indexing → internal `torch.nonzero` (`:2024`); 0-dim tensor as slice bound → `Tensor.__index__` → `.item()` (`:2034`); `if timestamp_logprob > max_text_token_logprob` → `__bool__` (`:2049`). Plus full-vocab `scores.clone()` (`:2007`) + fp32 log_softmax (`:2045`) per token. Instrumented at B=8: **~32 syncs/token**.
- **Default path:** long-form (>30 s mel) FORCES `return_timestamps=True` (`generation_whisper.py:1388-1397`; False raises); processor inserted first at `:1775-1778`. Always eager. Who pays: all podcast/meeting transcription, generate or ASR pipeline.
- **Fix (validated, ~40-60 lines, 1 file):** vectorized with row-level stats + slice-local `masked_fill_` (full-vocab masks were 2-3× SLOWER on CPU — trap avoided). **Bit-identical across 150 random generation states**; CPU parity at B≥8; zero GPU syncs. Est. gain: **5-15% tokens/s B=1, 15-40% B=8-32** (GPU run needed). Scripts: /tmp/wts_bench2.py.
- **Bundle:** hoist known-below-bar `SuppressTokensLogitsProcessor` arange+isin (`:1907-1910`) into a static mask in the same PR (Whisper applies it by default).
- **Duplicates:** clean. Open #43139 is feature-extractor-only; #43089 is generic generate syncs (Finding 21).

**12b (was W2) — `_dynamic_time_warping`: pure-Python O(M×N) loop — 57-245 ms per row per 30 s segment for word timestamps:**

- `generation_whisper.py:74-88`; called per batch row per segment from `_extract_token_timestamps` (`:352` → `:367`) when `return_token_timestamps=True` (= ASR pipeline `return_timestamps="word"`, the subtitle/alignment path). ~675K interpreted iterations per (448×1500) matrix; 1-hour file ≈ **10-30 s of pure Python DTW** while GPU idles.
- **Fix (validated, ~40 lines):** anti-diagonal wavefront vectorization in numpy — M×N scalar steps → M+N vector steps; tie-breaking replicated exactly; **bit-identical on 13 test cases; 3.3×/5.4×/6.6× at M=100/250/448**. End-to-end: ~5-20% wall-clock for word-timestamp transcription. Fully CPU-proven (/tmp/dtw_bench.py).
- **Adjacent (cite, don't collide):** open PR #33732 owns the memory half (attention weights retention); the eager-forcing (`:704, :1375`) is architectural.
- **Duplicates:** clean (#28114/#28226/#25607 are correctness fixes).

### Finding 13 (was P2) — Every multimodal processor call counts placeholder tokens via `list(tensor).count(int)` — 21-24% of real Qwen2.5-VL processor call

- `ProcessorMixin._check_special_mm_tokens` (`processing_utils.py:2332`): `[list(ids).count(token_id) for ids in input_ids]` — with pt tensors = one 0-dim tensor + `__eq__`/`__bool__` per token; runs unconditionally from `__call__` (`:691`), twice for Qwen (image+video tokens).
- **Measured (real Qwen2.5-VL-7B processor, call-counter proven):** image call 4.0/16.9 ms (24%), whole-call 1.56× with fix; video 7.0/33.2 ms (21%), 1.28×; 270-1100× on the primitive; bit-identical outputs. Also paid by vLLM/SGLang (they call HF processors per multimodal request).
- **Fix (2-4 lines):** `(ids == token_id).sum().item()` for tensors. **Duplicates:** none; adjacent open PR #43297 (SGLang, Qwen3VL placeholder re-tokenization — evidence this stage is a serving bottleneck; cite, don't collide). **Verify: CPU (done).**

### Finding 14 (was V1) — Eval loop: O(N²) per-batch logits re-concatenation + 2× transient peak + silent bf16→fp32 CPU inflation

**Status: general "eval accumulates logits" OOM is known (#7232, #28769, #32282); the quadratic/2× mechanics are UNREPORTED. Measured.**

- **Mechanism:** with `compute_metrics` set, `prediction_step` returns full `[B,S,V]` logits (`trainer.py:2624, 3021-3025`); defaults (`batch_eval_metrics=False`, `eval_accumulation_steps=None`, `eval_do_concat_batches=True`) → `all_preds.add(logits)` (`trainer.py:2771`) → `nested_concat` reallocates + copies the FULL accumulated buffer every batch (`trainer_pt_utils.py:377-378`). `eval_accumulation_steps` workaround does the same quadratic re-concat on CPU at each flush (`trainer_pt_utils.py:392-393`) and `nested_numpify` silently converts bf16→fp32 (`trainer_pt_utils.py:167-171`) = 2× CPU bytes.
- **Measured (CPU):** 200 batches × 4 MB → **84.3 GB copied, 5.3 s** (vs ~0 for list-append; clean 4× scaling from 100 batches). cat needs old+new simultaneously → OOM at ~17 batches instead of ~35 on 80 GB (Llama-3 vocab, seq 1024, B=8 = 2.1 GB/step).
- **Fix (~15 lines):** `EvalLoopContainer.add` appends to list, concat once at flush — the class already has this code path under `do_nested_concat=False`. Output identical.
- **Duplicates:** `nested_concat quadratic` = 0 results; `EvalLoopContainer` = 3 unrelated.
- **Benchmark:** fully CPU-verifiable (done); GPU peak via mocked prediction_step.

### Finding 15 (was G2) — TopP full-sorts 128-256K vocab every token although default TopK(50) just ran

**Status: no duplicate; merged precedent = #42248 (identical optimization for min-p); #9557 (2021) optimized masking but kept the full sort.**

- **Mechanism:** under sampling, `_get_logits_processor` always adds TopK (default 50, `generation/utils.py:1279-1282`) then TopP (`:1283-1286`) — mainstream instruct checkpoints ship `top_p<1` (Llama-3.1: 0.9; Qwen2.5: 0.8). Per token: `TopPLogitsWarper.__call__` (`logits_process.py:527-529`) full `torch.sort` of [B, V] + full softmax/cumsum/scatter — at a point where only 50 entries are finite. Sorting only the top-k values is **bit-identical** (softmax renormalizes over finite entries).
- **Measured (V=262144, CPU):** TopP path 8.1 ms/token (B=1), 23.4 ms (B=8) vs 0.44 ms for topk(50) = ~18-20×. vLLM measured ~20% throughput overhead from this exact pattern (vLLM PR #11394) and applies top-p on the top-k set / rejection sampling; llama.cpp same.
- **Fix (~30 lines):** fused TopK+TopP or TopP detecting prior masking → sort 50 elements, map cutoff via topk indices. Exact; low risk. Standalone TopP unchanged.
- **Gain:** 1-3% big-GPU decode; **5-20% CPU/MPS + small models**; approaches vLLM's ~20% only in large-batch serving.

### Finding 16 (was X4) — `EmbeddingParallel`: all-zero embeddings for last-shard tokens when vocab % tp_size ≠ 0

- Shard uses ceil-division (`get_tensor_shard`, `:397-399`); runtime hook reconstructs range as `rank * mod.weight.shape[0]` (`:1001-1003`) — wrong for the smaller last shard. V=100, tp=8: tokens 91-99 claimed by NO rank → zero vectors after all-reduce; tokens 63-71 get correct + garbage. Fully silent. Mainstream vocabs divisible by 8 → triggers: tp 3/5/6/7 or odd vocabs (GPT-2 50257). **Fix ~8 lines** (store full vocab, compute ceil shard size) or loud validation error. No duplicates.

### Finding 17 (was X5+X6) — RoPE bugs in `modeling_rope_utils.py`: longrope dead cache guard + yarn `truncate` ignored in nested configs

**17a (was X5) — longrope inv_freq cache guard is dead code — Phi-3/3.5/4 recompute inv_freq EVERY forward past original context (+ per-step H2D copy); latent UnboundLocalError:**

- `modeling_rope_utils.py:64-74`: guard checks `hasattr(self, f"{layer_type}_long_inv_freq")` = `"None_long_inv_freq"`; value stored as `"long_inv_freq"` (prefix=""). Never matches → `rope_init_fn` every forward when seq > original_max (incl. `torch.tensor(long_factor)` pageable H2D per decode step, `:540-543`); for future layer_type≠None models: UnboundLocalError at `:73`. Verified: 5 steps → 5 recomputes. Magnitude honest: tens of µs–low ms/token, but it's a *bug*. **Fix ~4 lines.** No duplicates.

**17b (was X6) — yarn `truncate` silently ignored in nested per-layer-type rope configs:**

- `_compute_yarn_parameters` reads everything from the layer-type sub-dict EXCEPT `truncate` from the outer dict (`modeling_rope_utils.py:450`). Nested `truncate: False` → silently True. Verified vs DeepSeek-V4 config (`modeling_deepseek_v4.py:109`): 9/32 inv_freq dims differ up to 103% relative. Latent today. **Fix: 1 line** — bundle with 17a (same file). Adjacent: #42189 added the field.

### Finding 18 (was F1) — Flash-attention: per-layer device-to-host sync in `_is_packed_sequence` (batch=1 decode)

- **Mechanism:** `modeling_flash_attention_utils.py:534-547` — for B=1 returns a 0-dim CUDA tensor (`.abs().sum().bool()`); runs unconditionally per layer at `:764`; coerced via `elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:` at `:794` → `Tensor.__bool__` → sync. Decode step position_ids shape [1,1]: packedness provably False (len-1 seq can't be packed) — 3 kernels + a sync per layer to compute a constant. Even for B>1 the arange+min kernels at `:544-546` launch per layer before short-circuit. In the padded branch (`if attention_mask is not None:` at `:770`) the result is computed and never used.
- **Preconditions verified:** every layer receives position_ids (`modeling_llama.py:415`); generate's all-ones mask is nulled once per forward at `masking_utils.py:651-652` (`flash_attention_mask`), so decode hits the sync branch. Function introduced in PR #40002 (Aug 2025).
- **Fix 1 (~10-20 lines, 1 file):** early-return False when `position_ids.shape[-1] == 1`; move the call out of the dead branch. **Gain: 5-15% tokens/sec small models, 2-5% 8B** (n_layers syncs/token eliminated).
- **Coordination:** issue **#39213** (OPEN since Jul 2025, zero maintainer replies, author inactive >1yr) describes the older form (`torch.diff(...).all()`) of this exact sync — comment there with benchmarks and take over. **PR #47342** (winglian/Axolotl, open, reviewers ArthurZucker+SunMarc) fixes 1D/3D position_ids shapes in this same function — rebase over it. **#46693** (closed): FA2 long-context perf regression escalated to Cyril → the `.item()` comment at `:369`; team is sensitive here.

### Finding 19 (was F2) — Flash-attention: `_get_unpad_data` recomputed per layer with 2 syncs (padded training) + `_prepare_from_posids` per layer (packed training)

- **Mechanism:** padded branch → `_upad_input` per layer → `_get_unpad_data` (`modeling_flash_attention_utils.py:367-371`): `attention_mask.sum` + `torch.nonzero` (documented synchronizing op) + `.max().item()` (sync) + cumsum — identical results per layer, ~2 syncs × n_layers per training step. Packed-without-collator path: `_prepare_from_posids` → `prepare_fa_kwargs_from_position_ids` (`:458-493`, nonzero sync + cu_seqlens rebuild per layer). `DataCollatorWithFlattening` DOES provide `cu_seq_lens_q/k` (`data/data_collator.py:1364-1438`) — mitigates only that collator's users. Famous old comment at `:440`: "There is a memcpy here, that is very bad."
- **Fix 2 (~150-300 lines):** hoist varlen metadata (indices, cu_seqlens, max_seqlen, packedness) to once-per-forward — same pattern #47170 did for vision models. Design decision (central cache vs per-model plumbing) → propose on #39213 first. **Traps:** hybrid sliding+full models have per-layer-type masks (cache must key per mask); 3D/1D position_ids (see #47342); StaticCache K/V slicing (`:424`); is_tracing/compile no-op; cross-attention q≠k branches. Acceptance bar: bitwise-identical outputs.
- **Blast radius:** ALL ~501 model families funnel through `_flash_attention_forward` via the single gateway `integrations/flash_attention.py` (FA2/3/4 + kernels-hub fallbacks + NPU/XPU). Continuous batching has its own paged path (not affected).
- **Gain:** 1-5% training step (padded fine-tunes); combined with Finding 18, "every FA model, training+inference".

### Finding 20 (was C2) — Padded-batch decode pays full `repeat_kv` materialization of the growing KV cache every layer/step

- One pad token in the batch → `(B,1,1,T)` mask every step → `use_gqa_in_sdpa` False → repeat_kv copies whole cache per layer per step for the entire generation. Verified by execution.
- Llama-3-8B B=4 T=4096: ~537 MB extra/layer → ~17 GB extra/token ≈ ~2× step time; T=1024 ≈ +25%.
- **Root cause is upstream:** SDPA `enable_gqa` is flash+math-only; flash can't take attn_mask (pytorch#139586, #154363). Options: route to cuDNN backend (accepts mismatched head counts, version/hardware-gated) or push upstream mem-efficient GQA broadcast. Adjacent transformers work: #45636 (open proposal), #45776 (open PR). Gain if routed: 20-50% batched long-gen decode; fix confidence medium-low. **Stash: upstream-blocked.**

### Finding 21 (was C3) — Per-token GPU syncs in generate/masking — KNOWN, stalled fixes to revive

- Sites: (a) `fast_all(padding_mask)` bool-coerced in `_ignore_causal_mask_sdpa` (`masking_utils.py:268`; hybrid models sync twice) — recomputed every step on a growing (B,T) mask whose padding pattern is fixed after prefill → result cacheable per generation; (b) `unfinished_sequences.max() == 0` (`generation/utils.py:2920` → `:2677`); (c) one in `_cache_dependant_input_preparation`.
- ~1.5 ms/sync measured in the existing report; up to 2-3× per-token latency for <1B models; 5-15% for 7-8B eager.
- **Duplicate status: KNOWN — issue #43089 (closed by stale-bot) + open unmerged PRs #43085 (async stopping criteria) and #43088 (skip attention_mask.all() sync).** Action: revive/support those PRs, don't open new ones. Good reviewer-rapport move.

### Finding 22 (was G3) — Continuous-batching streaming O(N²) on producer AND consumer sides

- Per generated token: `state.to_generation_output()` (`continuous_api.py:456-457`) copies full history (`requests.py:289-295`); full-history GenerationOutput then routed per token through `OutputRouter.deliver` (`continuous_api.py:97-105`). Quadratic cost lands on the single batch-manager thread shared by all requests.
- **Measured:** 8K tokens: 43 ms copying vs 0.23 ms append-only = 185×; 32K reasoning output ≈ 2.7 s pure copying/request; 50 concurrent × 8K ≈ ~2 ms/decode-iteration stolen from scheduling.
- Fix (~20 lines): delta delivery via `tokens_sent` cursor + final full output. Changes streaming payload contract → serving consumers need matching update. Gain: 10-30% scheduler-thread headroom in long-output serving. No duplicate found. CPU-benchmarkable (mock model + py-spy).

### Finding 23 (was G4) — Beam search: full-cache `index_select` copy every step even for identity beam_idx

- `_beam_search` (`generation/utils.py:3462-3468`) → `reorder_cache` per layer (`cache_utils.py:99-103`); plus `_flatten_beam_dim(running_sequences...)` copy per step (`utils.py:3349`). num_beams=4, 8B, T=2048: ~2.1 GB read+write/step on top of cat growth. Fix: skip when `beam_idx == arange` (fold comparison into existing stopping sync) or lazy indirection. Gain 10-30% of beam decode — but beam search is a minority path in 2026. #46672 fixed correctness here, not perf.

### Finding 24 (wave 5, discovered Jul 29 2026 — post-renumbering; would slot at #2–3 by impact) — Gemma-2 / VaultGemma / T5Gemma silently drop attention softcapping on the DEFAULT sdpa backend — 19-month regression of #32865, deleted by #35235

**Silent wrong outputs on a flagship Google model's default path, invisible to the entire test suite, with a clean git-archeology story. Highest review-impact class (silent correctness, not perf).**

- **Root cause:** `sdpa_attention_forward` (`integrations/sdpa_attention.py:79-90`, served for `"sdpa"` via `modeling_utils.py:5126`) has NO `softcap` parameter. Gemma2 passes `softcap=self.attn_logit_softcapping` (=50.0 in the config *default* and all hub checkpoints) into the interface call (`modeling_gemma2.py:293`; attr set at `:256`) → swallowed by `**kwargs`, silently ignored. Default resolution is sdpa (`modeling_utils.py:1940`); nothing in the dispatch machinery checks softcap (grep-verified, zero mentions in core). FA2/3/4 and flex implement softcap natively (`modeling_flash_attention_utils.py`, `integrations/flex_attention.py`) — only sdpa (and `paged|sdpa`) drop it.
- **Regression arc (verified via git):** #31698 (Jun 2024, "Gemma capping is a must for big models" — softcap is load-bearing, logits explode past 50 without it) → #32865 (Aug 22 2024, Joao Gante, "Gemma2: eager attention by default": overrode `_check_and_enable_sdpa` to force sdpa→eager by default + regression test; quote: *"SDPA reduces the model performance on Gemma2 because of the logits softcapping"*) → #35235 "🚨All attention refactor🚨" (merged Dec 18 2024) **deleted the override and the test, no replacement** (`git log -S _check_and_enable_sdpa`: only these two commits). Window: v4.48 → v5.8 (~19 months), every default-path gemma-2 load.
- **Executed verification (all CPU):** (1) bitwise: sdpa fn with vs without `softcap=50.0` → diff exactly 0.0 (kwarg swallowed). (2) default `Gemma2ForCausalLM(cfg)` resolves to `"sdpa"`; real checkpoint config (unsloth/gemma-2-9b mirror of the gated google repo): attn 50.0 / final 30.0, no attn_implementation field → default resolution applies. (3) effect size, causal mask equalized on both backends so softcap is the ONLY difference: max softmax-row L1 shift 0.038 (attn-logit std≈4, random-init regime) / 0.288 (15) / 0.595 (30) / **1.925 (60, trained regime)**. Control with softcap=None both sides: 5e-7.
- **Why tests miss it:** at random init, attention logits ~O(1) and tanh(x/50)·50 ≈ x to ~4 decimals; divergence only materializes at trained-magnitude logits (std ≥15–60) — tiny-config CI can never see it. (A model-level eager/sdpa A/B at random init prints 0.000000 even with q/k amplified — init regime; the L1 table isolates the true effect. Do not be fooled by it at repro time.)
- **Who pays:** gemma-2-2b/9b/27b + all derivatives (`_supports_sdpa=True`, `modeling_gemma2.py:371`); vaultgemma (config default 50.0, passes softcap `modeling_vaultgemma.py:225`, flags `:361-363`); t5gemma (default 50.0, softcap at `:309` encoder + `:395` decoder, flags `:562-564`). Unaffected: gemma3/3n/4 (attn softcap None / final-logit-only = backend-independent), flash/flex users. Install base: every stock install without flash-attn (CPU, MPS, default pip) — the majority.
- **Precedent proving intent:** videoprism sets `_supports_sdpa = False` (`modeling_videoprism.py:480`) explicitly for attn capping — the codebase knows the invariant; these three models lost theirs.
- **Fix (~15 lines + tests):** (1) structural guard — `sdpa_attention_forward` takes `softcap=None` explicitly and warns/raises loudly when non-None (kills the class for all future models); (2) restore #32865 semantics via the v5 dispatch hook — gemma2/t5gemma/vaultgemma refuse/fallback sdpa when softcap set (videoprism precedent); (3) restore the #32865 regression test (default resolution ⇒ eager for softcap configs) + interface-guard test.
- **Duplicates:** clean. GitHub "softcap sdpa" → 2 unrelated (FA4 PR #42404, is_causal issue #39554); "gemma softcap" → 7 (2024 fix-era #32309/#34282, GGUF mapping draft #42881, #45676 gemma-4 grad spikes closed-unrelated, videoprism #39895) — none report the current state.
- **Verify:** fully CPU (done). Optional GPU demo: real gemma-2-9b eager-vs-sdpa sample/perplexity diff for the headline chart.
- **Action:** new issue (correctness regression) citing #31698/#32865/#35235 with the L1 table + bitwise demo; PR = interface guard + per-model protection + tests. Tag ArthurZucker (authored #34282 gemma capping) + Cyril (attention interface owner). Coordinate on issue first per policy.

---

# SUPPORTING MATERIAL

## Cyril PR anatomy (distilled from deep web research — reference for framing all PRs)

- **#30536**: NOT just "cat copies" — a REFERENCE-LIFETIME bug: per-step `to_legacy_cache()`/`from_legacy_cache()` round-trip held refs to pre-cat tensors → GC couldn't free them → peak = 2× live data. Fix = data-structure change + reference hygiene (`del outputs`, clone `next_token_logits`). Benchmark style: memory-vs-length curves, one line per commit.
- **#31292**: computing strictly more than consumed (`[B,S,V]` logits at prefill when `[B,1,V]` read) → `num_logits_to_keep` plumbed everywhere, slice BEFORE lm_head.
- **#33570**: condition order in an `and` chain evaluated a GPU tensor `.all()` before cheap Python checks → per-step sync. Fix = reorder for short-circuit.
- **#37866/cache refactors/loading rework**: redesign instinct — make the waste class *impossible to write*, not just fixed once.
- Common shape: (a) object-lifetime bugs, (b) slice-suffices waste on default paths, (c) torch semantics contradicting code intent, (d) structural redesign. Landing formula: flagship-model default path + arithmetic + reproducible curve + minimal diff.

## Below the bar (checked; don't PR alone)

**Wave 1:**
- `inspect.signature` per processor per token (`logits_process.py:87`): 4.5 µs/call → ~15 µs/step. Bundle-only.
- Full-vocab fp32 logits copy per sampling step (`generation/utils.py:2877`): ~1 MB/step B=1 → ~12 µs; load-bearing for sampling numerics.
- Rotary cos/sin recomputed per forward (`modeling_llama.py:408`): ~6 small kernels/step, micro-noise.
- Whisper/suppress_tokens rebuilds arange+isin per step (`logits_process.py:1907-1910`) — niche (bundled into Finding 12a).
- Streamer `.cpu()` per token (`generation/utils.py:2914-2917`) — known, small.
- Serving `apply_chat_template` sync on asyncio loop (`cli/serving/chat_completion.py:108-150`) — serving-only.
- Image `_preprocess` double group→stack→regroup (`image_processing_backends.py:387-407`); CPU default disables batched torchvision ops (`image_transforms.py:1014-1025`) — vision-niche.
- MoE eager fallback syncs (`modeling_qwen3_moe.py:227-251` nonzero+tensor-compare per expert; `integrations/moe.py:202` `offs.tolist()`) — only CPU/torch≤2.8/pre-SM80/MPS; default `grouped_mm` path (`moe.py:383-484`) is clean.
- `tie_weights` `torch.equal` on full embeddings (`modeling_utils.py:2655-2657`); `resize_token_embeddings` fp32 full copy (`:3113-3116`) — one-off costs at load.

**Wave 2 (bundle candidates only):**
- `get_rope_index` prefill: per-row Python loop + 1-2 syncs/row + groupby — once per generate, ~ms.
- `_deepstack_process` (qwen3_vl `:856-862`): full clone + nonzero sync ×3 deepstack layers, prefill only.
- `get_placeholder_mask` `torch_compilable_check` → 1 sync per multimodal prefill (llava:209, qwen2_5_vl:1148, gemma3:820).
- `apply_rotary_pos_emb_vision` fp32 upcast per vision layer — bandwidth noise.
- Llava forced `output_hidden_states=True` retains ~25 CLIP states (~30-150 MB/image) — KNOWN: #33698/#33705.
- `tokenizer.pad` tensor-features `to_py_obj` round-trip (4.26 vs 3.34 ms) — absorbed by Finding 7.
- PilBackend CHW→PIL→CHW double transpose — non-default backend.

## Audited and CLEARED / REJECTED (don't re-spend time)

**Wave 1:**
- **v5 `from_pretrained` loading** (`modeling_utils.py:4410-4513` → `core_model_loading.py:1446-1737`): lazy safetensors slices, fused per-tensor `.to(device,dtype)`, 4-thread pool, allocator warmup. Residual nit: pageable H2D (maybe 20-30% of load time) — recently reworked by HF, low-value target.
- **`save_pretrained`** (`modeling_utils.py:3468-3573`): no redundant full CPU copy on default path.
- **Gradient checkpointing**: `use_reentrant=False` default (`modeling_utils.py:3214`) — efficient variant.
- **`num_items_in_batch`** (`trainer.py:2140-2200`): CPU-side, no sync. `tr_loss` accumulation on-device.
- **Prefill logits**: `logits_to_keep=1` set at `generation/utils.py:2598` — #31292-class issue is fixed.
- **optimization.py schedulers, data_collator defaults**: nothing per-step of consequence.

**Wave 2:**
- **QuantizedCache error compounding: REJECTED experimentally** — token-0 key bit-stable across 300 steps (axis=0 per-token groups round-trip losslessly). Only known-slow perf angle remains; below bar.
- `crop(0)` empties sliding cache — real footgun, unreachable from generate; #47433 owns it.
- Chat templates (Jinja lru_cached), `get_text_with_replacements` (O(n)), pipelines base, fast-tokenizer `_encode_plus`, TorchvisionBackend fused rescale+normalize: clean.
- Vision tower NOT recomputed during decode (pixel_values gated to first iteration — llava `:401-407`, qwen2_5_vl `:1483-1486`, gemma3). Per-step M-RoPE O(1) (`generation/utils.py:947-957`). Whisper (batched stft, cross-attn KV cached via is_updated). SigLIP/CLIP/Gemma3 towers single batched SDPA. Video preprocessing groups+batches.
- masking_utils q_offset/skip/packed-detection consistent with cache semantics (only Findings 3/20 remain).
- PEFT integration + hub_kernels: clean.

**Wave 3:**
- T5-family eager-only attention: real gap, **claimed by open PR #47014** (+#31167/#42453). Dead for us.
- WhisperFeatureExtractor hann/mel per-call: claimed by open PR #43139.
- `gelu_new` unfused (~7 kernels): deliberate bit-compat (GELUTanh docstring); hub-kernels opt-in exists. Would be rejected.
- `EosTokenCriteria` `.to(device)`: one-time (assigns back). Trainer grad-norm/`num_input_tokens_seen`: clean on defaults. v5 decorator stack: µs/token (cProfiled). T5 `compute_bias` per step: small, belongs to #47014. wav2vec2 `_compute_mask_indices`: ms-scale niche. GPT-OSS eager-with-sinks: architectural, #41465 owns it. Whisper seek-loop `.item()`s: per-30s-segment, fine. Qwen2.5-VL `_expand_inputs`: beam-only, once per generate. NoBadWords `.tolist()`: init-time. BERT/RoBERTa: modernized, clean. Per-step input_ids/mask cat: dwarfed by Finding 4, fold into its narrative.
- Note: /tmp bench scripts need `lzma` stub workaround (pyenv python lacks `_lzma`) + `sys.path.insert(0, ".../src")`.

**Wave 4 (killed candidates):**
- DeepSeek-V3 MLA expanded-KV cache (~71× latents): settled territory — #47460 merged Jul 23 2026 deliberately keeps expansion (vLLM bypasses); #47011 closed unmerged. Do not relitigate.
- `_sample` logits lifetime: already clean (Cyril's own `del outputs` + `.to(copy=True)`).
- Assisted decoding retaining outputs: intentional (candidate generator needs them; `crop()` bounds memory).
- New StaticCache per generate call: dead — persisted on `self._cache`.
- `fast_all` sync (= Finding 21 site a), repeat_kv-under-mask (= Findings 3/20): already filed above.

## Key torch reference facts (verified)

- `Tensor.__bool__`/`item()`/`.cpu()`/`torch.nonzero` on CUDA tensors force host-device synchronization (PyTorch CUDA-semantics docs; the top item in the perf-tuning guide's list of offenders).
- Python `and`/`or`: `A and B` short-circuits without evaluating B's truthiness if A falsy; returns B *unevaluated as bool* if A truthy — so a returned 0-dim tensor syncs at its first boolean *use*, not at creation.
- SDPA `enable_gqa`: flash+math backends only; flash backend cannot take `attn_mask` (pytorch#139586, #154363). cuDNN backend natively accepts mismatched head counts.
- `expand().reshape()` on the expanded dim forces a materialized copy (`repeat_kv` pattern).
- F.cross_entropy/log_softmax for half/bf16 inputs accumulate internally in fp32 (AccumulateType; PyTorch PR #24457) — the `.float()` upcast buys fp32 storage of log-probs, not fp32 math.
- PyTorch caching allocator handles monotonically growing allocation sizes poorly (freed S-blocks can't serve S+1 requests → fragmentation).

## GitHub coordination map (all findings)

| Finding | Vehicle | Urgency |
|---|---|---|
| 1 (SWA assisted crash) | New issue w/ CPU repro + ~8-line PR referencing #47447 | **IMMEDIATE — pre-release window** |
| 2 (Qwen video M-RoPE) | New issue + PR w/ reference-diff table; cite #37597/#43972/#45400 lineage | High |
| 3 (sliding-window mask) | NEW issue+PR — no prior art; cite #37866 as origin of the condition | High |
| 4 (DynamicCache prealloc) | NEW issue first (design discussion: views vs contiguity); cite #30536 lineage | High |
| 5 (Trainer sync) | NEW issue+PR — unreported | High (first PR) |
| 6 (static SWA roll) | NEW issue (not #44609 — compile-topic, closed) w/ bandwidth arithmetic + eager/compiled benchmarks | High |
| 7 (BatchEncoding convert) | New PR; cite #33855 (maintainer buy-in) + BatchFeature precedent | High |
| 8 (video loading crash + seek) | Issue+PR for 8a crash now; 8b as follow-up commit in same thread | High |
| 9 (TP save corruption) | One issue "TP save corrupts checkpoints" + one PR for both 9a+9b | High |
| 10 (chunked CE) | Coordinate — Liger-integration discussion territory (~#34191 era), prior #33902/#34147 history | Medium |
| 11 (vision tower windows) | Stage 1 sync-hoist mini-PR → Stage 2 batched-windows PR citing #37363's own numbers | Medium |
| 12 (Whisper timestamps) | Single mini-campaign; cite #33732 (memory half) as adjacent | Medium |
| 13 (mm token count) | Tiny PR; cite open #43297 as adjacent evidence, don't collide | Medium |
| 14 (eval concat) | NEW PR — mechanics unreported; cite #7232/#28769 as context | Medium |
| 15 (TopP-after-TopK) | NEW PR; cite merged #42248 (min-p) as precedent | Medium |
| 16 (embedding shards) | Same thread as Finding 9 or loud validation error | Medium |
| 17 (rope fixes) | One small PR in modeling_rope_utils.py (17a+17b) | Medium |
| 18/19 (FA syncs) | Comment on OPEN issue #39213 (stale >1yr, unassigned) with benchmarks; rebase over open PR #47342; cite merged #47170 as design precedent | Medium |
| 20 (padded repeat_kv) | Upstream-blocked; watch #45636/#45776 | Stash |
| 21 (generate syncs) | Revive stalled open PRs #43085 / #43088 (issue #43089, stale-closed) | Stash |
| 22 (CB streaming) | New issue when serving focus | Stash |
| 23 (beam reorder) | Bundle-only | Stash |
| 24 (sdpa softcap drop) | New issue (regression; cite #31698/#32865/#35235 w/ L1 table + bitwise demo) + PR: interface guard + per-model sdpa refusal + restored #32865 test | High |

## Wave-5 additions (Jul 29 2026 — minor findings from the same sweep, not renumbered)

- **N-1. `NoRepeatNGramLogitsProcessor`: O(T²) python + B full-prefix `.tolist()` syncs per token** (`generation/logits_process.py:1041,1069,1136-1144`) — rebuilds the entire n-gram dict from scratch per row per token. Measured: 4.1 ms/token at B=8 T≈1280; 1.6 ms/call at T=4096 B=1. Default-on for facebook/bart-large-cnn (hub generation_config: `no_repeat_ngram_size:3, num_beams:4`) + standard anti-repetition flag. Fix ~30 lines: vectorized `unfold` scan for only the current (n−1)-gram, zero syncs, bit-identical banned sets.
- **N-2. EAGLE/MTP drafter: 2 avoidable syncs per drafted token** (`generation/candidate_generator.py:1392` `if sequence_stopped.any():` + `:1409` `.all()`; H2D `torch.tensor` per round at `:1368`). The `.any()` branch costs more than the `torch.where` it guards. ~10 lines. Niche (speculative-decoding users).
- **N-3. `create_recurrent_attention_mask` `torch.all(attention_mask == 1)` sync** (`masking_utils.py:1474`) — per recurrent layer on every q>1 forward (prefill/continuation) for hybrid linear-attn models (qwen3_next, jamba, zamba2, bamba, lfm2). Distinct site from Finding 21's `fast_all` (`:268`); unowned by #43085/#43088. Bundle with Finding 21 revival.
- **N-4. UAD different-tokenizer: per-round O(P·N) compare + sync + python diag scan** (`candidate_generator.py:456-476`). Niche.
- **N-5 (bundle-only):** gemma3 VLM loss bool-index sync/step (`modeling_gemma3.py:1042-1046`); gemma4 vision `split_sizes.tolist()` per image forward (`modeling_gemma4.py:2229,2503`).
- **Wave-5 dead ends (verified, don't re-spend):** `find_packed_sequence_indices` sync (training-packing only, once/forward); `is_attention_mask_causal` (context-parallel only, cached once); `_merge_kwargs` deepcopy (small option dicts); sdpa/flash integration files beyond Findings 3/18-20; `_update_model_kwargs_for_generation` cats (folded into Finding 4); `Trainer.log`; MoE eager `expert_hit` nonzero (non-default path, previously cleared); flagship LLM files (llama/qwen3/gemma3/mistral) free of per-forward H2D/syncs.

## PR-collision status (verified Jul 25, 2026)

> Fresh GitHub API sweep (search + PR status fetches), 2-3 keyword variations per finding, top-12 findings.
> "Unclaimed" = no open/closed/merged PR or issue addresses the mechanism. Merged PRs dated before Jul 23
> cannot invalidate findings — the local checkout `e9c987a3` (synced Jul 23) already contains them and all
> findings were verified on it.

| Finding | Our status | Relevant PR/issue #s (state, last activity) | Notes |
|---|---|---|---|
| 1 (SWA assisted crash) | **UNCLAIMED — window still open** | #47447 (merged Jul 22 — the regression source); #47433 (issue, OPEN, Jul 22 — crop(0), different bug, still unowned by a fix); #47153 (PR, closed unmerged, Jul 20 — restored dynamic_full guard for assisted decoding, pre-dates #47447) | Searched `47447` mentions + `get_mask_sizes` + crash keywords: only the PR itself and #47433 reference it. 3 days post-merge, nobody has reported the crash. ACT NOW. |
| 2 (Qwen video M-RoPE) | Unclaimed | Known lineage only: #45330 (merged Apr 2026, still-image temporal scaling), #41153 (merged Sep 2025), #42009 (merged Nov 2025, fps assignment) | `second_per_grid_ts` + mrope searches surface nothing on int-truncation / text-offset / delta divergences. |
| 3 (sliding-window mask) | Unclaimed; adjacent area ACTIVE | **NEW: #46960 (MERGED Jul 2, 2026)** — "Fix silent SDPA math-kernel fallback for GQA when head_dim > 256 or differ" — same `use_gqa_in_sdpa` gate, head_dim condition only; #45776 (PR, open DRAFT, May 6 — enable_gqa gating; 3 vasqu review rounds, Cyril cc'd, stalled ~2.5 mo); #45636 (issue, open, May 29 — sdpa_memeff proposal) | #46960 is precedent that maintainers accept GQA-gate fixes in this exact code; merged pre-checkout so our mask condition still stands. Cite it; expect churn in `sdpa_attention.py`. |
| 4 (DynamicCache O(T²)) | Unclaimed | #40328 (RFC PR, open, last upd Nov 2025 — compile+DynamicCache, orthogonal) | `DynamicCache preallocate` / `cat quadratic` searches: zero claims. |
| 5 (Trainer NaN-filter sync) | Unclaimed | #30465 (merged Apr 2024 — added `non_blocking` *support* to accelerate dataloaders; default still False = our mechanism B) | `logging_nan_inf_filter` has 42 hits, all unrelated (config dumps in bug reports); nothing recent, nothing perf. |
| 6 (static SWA roll) | Unclaimed | #45745 (PR, open, Jun 22 — StaticCache crop(); **CHANGED: Cyril reviewed Jun 22: "I don't think this is worth the effort as it is in the current state"** — effectively dying, orthogonal anyway); #45846 (PR, closed unmerged, May 14 — snapshot/restore); #45940 (PR, closed unmerged, May 15 — get_mask_sizes 1-liner, rejected); #44609 (issue, closed Mar 12) | No ring-buffer/bandwidth work anywhere. Cyril's #45745 pushback confirms: motivation+numbers must be airtight. |
| 7 (BatchEncoding convert) | Unclaimed | #33855 (issue, closed stale Nov 2024 — maintainer "feel free to open a PR"); #32101 (issue, closed May 2026 — "Trainer + pretrained tokenizer + 4D attention mask extremely slow", extra motivation evidence) | No PR ever delivered the conversion fix. |
| 8 (video loading crash) | Unclaimed | #38880 (merged Jun 2025 — added torchcodec support; context, not the fallback fix) | No hits for the torchvision ≥0.26 crash or backend fallback chain; seek-based decoding (8b) also unclaimed. |
| 9 (TP save corruption) | Unclaimed — **but WATCH #47194** | **NEW: #47194 "TP dtensor" (PR, open DRAFT, upd Jul 10, 0 reviews)** — TP rework that may rewrite the save path; #45028 (merged May 2026, FSDP+TP native save/load — pre-checkout, our repro stands on top of it); #45662 (PR, open, May 13 — EP+FSDP2, different bug) | `RowwiseParallel bias state_dict` = 0 results. Action: inspect #47194's diff before filing; if it touches `gather_state_dict_for_save`, file the issue anyway (correctness) and flag the overlap in it. |
| 10 (chunked CE) | Known territory (unchanged); coordinate | **NEW: #38452 (issue, OPEN, upd May 2025)** — "Memory saving by upcasting logits for only non-ignored positions" — adjacent upcast-memory discussion, good coordination vehicle | No chunked-CE PR in core. Liger-style territory still unclaimed in-repo. |
| 11 (vision tower windows) | Unclaimed | #45396 (merged May 2026 — extracted dynamic vision tensors into pure functions; same region, pre-checkout, `.tolist()` sync still present on `e9c987a3`); #45816 (PR, closed unmerged, May 7 — Omni vision compile graph breaks); #47170 (merged Jul 21 — hoisting precedent); #45750 (issue, closed Jun 15 — Conv3d patch embed, different) | Serialized-window-SDPA + sync-hoist still unclaimed. |
| 12 (Whisper timestamps) | Unclaimed | #43139 (PR, open DRAFT since Jan 7, 0 reviews — "[perf] optimize whisper GPU performance", feature-extractor scope per audit; stalled ~6 mo); #33732 (PR, open, last upd Dec 2024 — DTW attention-weights memory half, stale) | Neither touches the per-token syncs (12a) or Python DTW loop (12b). Skim #43139's diff once more before filing to confirm scope. |

**Adjacent-PR status snapshot (fetched Jul 25, 2026):**

| PR | State | Last activity | Reviews / notes |
|---|---|---|---|
| #47342 (FA position_ids shapes) | open | Jul 16 | Reviewers requested: ArthurZucker, SunMarc; 0 reviews submitted yet. Still rebase-over target for Finding 18. |
| #43297 (Qwen3VL pad re-tokenization) | open | Mar 18 | 7 review comments (zucchini-nlp, itazap), stalled ~4 mo. Cite for Finding 13, still no collision. |
| #45745 (StaticCache crop) | open | Jun 22 | **Cyril skeptical Jun 22 ("not worth the effort as-is")** — see Finding 6 row. |
| #43085 (async stopping criteria) | open | Feb 9 | 0 reviews, idle ~5.5 mo — revival play for Finding 21 still available. |
| #43088 (skip mask.all() sync) | open | Jan 23 | 6 review comments (vasqu, zucchini-nlp engaged, then stalled) — revival play for Finding 21 still available. |
| #33705 (hidden_states index selection) | closed unmerged | Apr 2025 | Llava hidden-states waste (below-bar item) remains open territory but was previously attempted+abandoned. |
| #47014 (T5 SDPA) | open, ACTIVE | **Jul 24 (yesterday)** | 30 reviews; reviewers ArthurZucker, zucchini-nlp. T5 territory stays dead for us — confirmed. |
| #43139 (Whisper GPU perf) | open DRAFT | Jan 7 | 0 reviews — see Finding 12 row. |
| #47194 (TP dtensor) | open DRAFT | Jul 10 | 0 reviews — see Finding 9 row. |
| #45776 (enable_gqa gating) | open DRAFT | May 6 | 3 vasqu review rounds, Cyril cc'd, stalled — adjacent to Findings 3/20. |
| #46960 (SDPA GQA head_dim fallback) | **merged Jul 2** | Jul 3 | New precedent for Finding 3's gate area. |
| #47433 (crop(0) issue) | open | Jul 22 | Still an issue with no fix PR; still someone else's — don't touch. |

**Changed since the audits (action items):**
1. Finding 1 remains unclaimed 3 days after #47447 merged — the pre-release window is still open but this is the perishable one. File TODAY.
2. #46960 merged (Jul 2) inside the `use_gqa_in_sdpa` gate — new precedent + recent churn in Finding 3's exact fix site; cite it and verify line numbers at PR time.
3. #47194 (TP dtensor rework, draft) appeared Jul 10 — inspect its diff before filing Finding 9; possible save-path rewrite in flight.
4. Cyril pushed back on #45745 (Jun 22) — recalibrates Finding 6: no competing static-cache work landing soon, but the motivation bar is explicitly high.
5. #38452 discovered as an open coordination vehicle for Finding 10 (upcast memory discussion).
6. #47014 (T5 SDPA) very active as of Jul 24 — T5 stays cleared, no change needed.

## Immediate next actions

1. **Finding 1 first** — file issue with CPU repro + the ~8-line fix as PR; time-sensitive (pre-release, fresh in Cyril's context).
2. Build benchmark scripts (monkeypatch A/B/A + profiler sync counts + correctness gates) for Findings 5, 3, 4.
3. One Colab L4/A100 session: Findings 5, 3, 4, 18, 11-stage1. (Free T4 insufficient for FA2/Finding 18.)
4. CPU-only proofs already sufficient for: Findings 1, 2, 7, 13, 8 (a+b), 14, 9 (a+b), 17 (a+b).
5. User must be able to explain every mechanism in review; disclose AI assistance per repo policy; coordinate on issues before PRs.
