# BOICL for Cubic MoC Phase Isolation

**Bayesian Optimization with In-Context Learning (BOICL) applied to synthesis optimization of cubic molybdenum carbide (α-MoC) via Mo–sucrose carburization.**

This repository contains the full experiment campaign — datasets, notebooks, acquisition logic, and XRD reference data — for actively learning optimal carburization synthesis procedures to maximize the cubic MoC (Fm3̄m) phase weight fraction.

---

## Overview

Synthesis of cubic MoC is sensitive to a large combinatorial space of process parameters (temperature, ramp rate, hold time, gas composition, and precursor ratio). Rather than exhaustive search, this project uses **BOICL**: a GPT-4o-based surrogate that reads procedure text, predicts phase composition, and proposes new experiments — all via few-shot in-context learning, with no explicit structure-property model.

Two parallel acquisition strategies run as independent trajectories:

| Trajectory | Acquisition | Strategy |
|---|---|---|
| **T1** | Expected Improvement (EI) | Balances exploration and exploitation |


Each suggested procedure is physically synthesized and characterized by powder XRD with Rietveld refinement, and the result is fed back to the model for the next iteration.

---

## Key Results

- Best observed cubic MoC weight fraction: **96.5%** (T1)
- Trajectory converges toward the 620 °C / 0.5 hr / N₂ region of the design space

---

## Experimental Parameters

The following synthesis parameters are varied across the design space:

| Parameter | Values |
|---|---|
| Ramp rate | 5, 10, 15 °C/min |
| Carburization temperature | 550–900 °C |
| Hold time | 0.5, 2, 5, 10 hr |
| Gas | H₂ or N₂ |
| Flow rate | 30, 60, 100 sccm |
| AMT : sucrose ratio | 1:0.5, 1:1, 1:2 (g/g) |

Fixed conditions: 1 g ammonium heptamolybdate tetrahydrate, 120 °C drying (24 h), 212 µm sieve, 0.6 g loaded sample, N₂ cooling, 1% O₂/N₂ passivation.

---

## Repository Structure

```
boicl_crystal_phase_isolation/
│
├── cubic_WC_boicl.ipynb              # Main campaign notebook (T1)
├── bo_icl_env.yml                    # Conda environment
├── .env                              # API keys (git-ignored)
│
├── boicl/                            # Core BOICL package
│   ├── asktell.py                    # AskTellFewShotTopk — main BO interface
│   ├── asktellGPR.py                 # Gaussian process variant
│   ├── asktellRidgeRegression.py     # Ridge regression variant
│   ├── pool.py                       # Candidate pool with FAISS + OpenAI embeddings
│   ├── aqfxns.py                     # Acquisition functions (EI, PI, UCB, greedy)
│   ├── llm_model.py                  # LLM wrapper (OpenAI / Anthropic)
│   └── tool.py                       # LangChain tool integration
│
├── dataset/
│   ├── Metal_Sucrose_DesignSpace_fr_T1.xlsx   # T1 design space + labeled results
│   └── synthetic_Mo-Carburization_Dataset_v2_labeled.xlsx  # 2,500-procedure synthetic benchmark
│
├── campaigns/
│   ├── Synthetic/                    # Offline and live synthetic benchmark results (CSV)
│   ├── Trajectory_1_with_seeds_EI_acq/   # Raw XRD .dat files for T1 experiments
│   
│
├── BOICL_docs/
│   ├── pred_system_message.txt       # Forward model system prompt
│   ├── inv_system_message.txt        # Inverse design system prompt
│   └── inv_system_message_syn_data.txt
│
├── xrd/
│   ├── smartlab_Si640g.instprm.bak   # Rigaku SmartLab instrument parameters
│   └── Reference cif/               # CIF files for Rietveld refinement
│       ├── MoC_cubic_fixed.cif       # Target phase (Fm3̄m)
│       ├── MoC_0.66–0.75.cif         # Substoichiometric variants
│       ├── Mo2C_Pbcn_fixed.cif
│       ├── Mo_Im3m_fixed.cif
│       ├── MoO2_P21c_fixed.cif
│       └── graphite_2H_P63mmc.cif
│
└── archive/
    ├── chemistry_embeddings.ipynb    # Embedding model comparison (ada-002 vs 3-large vs MatBERT)
    └── BO_experiments.ipynb          # Earlier campaign iteration
```

---

## How It Works

### Forward Model (Predictor)

GPT-4o is prompted as a materials scientist specializing in Mo-carbide chemistry. Given a synthesis procedure string, it outputs a single number: the predicted cubic MoC weight fraction (0–100%). Few-shot examples are selected from the labeled pool using semantic similarity (FAISS + `text-embedding-3-large`), so the most chemically relevant prior experiments are always in context.

### Inverse Model (Designer)

A second GPT-4o call runs an inverse design: given a target MoC weight fraction, it proposes a new synthesis procedure. This can generate candidates outside the existing design space, augmenting the pool when the acquisition function identifies a region not covered by labeled examples.

### Acquisition

At each iteration, the model predicts MoC wf for every unlabeled candidate in the pool:

- **EI (T1)**: selects the candidate maximizing expected improvement over the current best

The chosen procedure is sent to the lab, synthesized, characterized by XRD, and the result is fed back via `asktell.tell(procedure, moc_wf)`.

### Pool Management

Candidates are embedded with `text-embedding-3-large` and indexed in FAISS. The `approx_sample` method uses greedy Max-Marginal-Relevance (MMR) re-ranking to select few-shot examples that are both relevant to the query and mutually diverse.

---

## Setup

### Environment

```bash
conda env create -f bo_icl_env.yml
conda activate bo_icl
```

### API Keys

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=sk-...
```

The model used for both prediction and embedding is `gpt-4o` / `text-embedding-3-large` via the OpenAI API.

### Pre-built Pool (optional)

To skip re-embedding the full design space on first run, place these files in the repo root:

```
mo2c_sucrose_prompts.pkl    # list of procedure strings
mo2c_sucrose_emb.npy        # (N, 3072) float32 embedding matrix
mo2c_sucrose_index.faiss    # FAISS IndexFlatIP
```

If absent, the pool will be re-embedded automatically on first instantiation and cached to `.emb_cache/`.

---

## Running the Campaign

Open `cubic_WC_boicl.ipynb` in JupyterLab and run sections in order:

1. **Trajectory 1 (EI)** — Cells 1–16
   - Update `completed_prompt_labels` in Cell 6 with new experimental results
   - Run through Cell 13 to get the next EI suggestion
   - Cell 16 renders the campaign summary table

---

## Phase Characterization

Raw powder XRD patterns (`.dat` / `.txt`) in `campaigns/` are refined using GSAS-II with the CIF files in `xrd/Reference cif/`. The Rietveld refinement reports weight fractions for:

- **α-MoC (Fm3̄m)** — target cubic phase
- **β-Mo₂C (Pbcn)** — hexagonal carbide (byproduct)
- **Mo (Im3̄m)** — unreacted metal
- **MoO₂ (P2₁/c)** — oxide (byproduct under oxidizing conditions)
- **Graphite** — excess carbon

The cubic MoC weight fraction is the optimization target.

---

## Embedding Model Comparison

`archive/chemistry_embeddings.ipynb` benchmarks three embedding models on their ability to discriminate synthesis conditions for Mo-carburization procedures:

| Model | SNR (chem/surface) | Unit sensitivity | Gas ranking |
|---|---|---|---|
| `text-embedding-ada-002` | **3.40** | 0.40 | 1/10 |
| `matscibert` | 2.31 | 0.56 | 5/10 |
| `text-embedding-3-large` | 1.48 | 0.79 | 5/10 |

`ada-002` has the highest signal-to-noise ratio for chemical changes vs. surface text noise. All models show poor unit sensitivity (< 1), meaning numerical parameter differences are underrepresented in embedding space — an open problem for future fine-tuning or contrastive learning approaches.

---

## Dependencies

Key packages (see `bo_icl_env.yml` for full pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `openai` | 1.77.0 | LLM API + embeddings |
| `anthropic` | 0.45.2 | Claude API (optional) |
| `faiss-cpu` | 1.9.0 | Vector similarity search |
| `langchain` | 0.3.17 | Prompt chaining + tool use |
| `botorch` / `gpytorch` | 0.12.0 / 1.13 | GP regression variant |
| `transformers` | 5.0.0 | MatBERT / HuggingFace models |
| `scikit-learn` | 1.6.1 | Cosine similarity, scalers |
| `numpy` / `scipy` | 1.26.4 / 1.15.1 | Numerics |
| `pandas` / `openpyxl` | 2.2.3 / 3.1.5 | Dataset I/O |

---

## Citation

If you use this code or dataset, please cite the associated work (forthcoming).
