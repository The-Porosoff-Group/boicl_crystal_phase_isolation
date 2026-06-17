# BOICL for Cubic MoC Phase Isolation

**Bayesian Optimization with In-Context Learning (BOICL) applied to synthesis optimization of cubic molybdenum carbide (α-MoC) via Mo–sucrose carburization.**

This repository contains the experiment campaign, datasets, notebooks, acquisition logic, XRD reference data, and analysis artifacts for actively learning carburization synthesis procedures that maximize the cubic MoC (`Fm3̄m`) phase weight fraction.

---

## Overview

Synthesis of cubic MoC is sensitive to a combinatorial space of process parameters, including temperature, ramp rate, hold time, gas composition, flow rate, and precursor ratio. Rather than exhaustively searching this space, this project uses **Bayesian Optimization with In-Context Learning (BOICL)**: a GPT-4o-based surrogate model that reads synthesis procedure text, predicts phase composition, and proposes new experiments using few-shot in-context learning.

The main experimental campaign uses a single active-learning trajectory:

| Trajectory | Acquisition               | Strategy                              |
| ---------- | ------------------------- | ------------------------------------- |
| **T1**     | Expected Improvement (EI) | Balances exploration and exploitation |

Each suggested procedure is physically synthesized and characterized by powder XRD with Rietveld refinement. The observed cubic MoC weight fraction is then fed back into the optimization loop.

---

## Key Results

* Best observed cubic MoC weight fraction: **96.5%** in the T1 campaign
* The trajectory converges toward a low-temperature N₂ carburization region near **620 °C / 0.5 hr / N₂**
* The repository includes raw XRD campaign files, Rietveld reference CIFs, campaign plots, and BOICL benchmarking artifacts

---

## Experimental Parameters

The following synthesis parameters are varied across the design space:

| Parameter                 | Values              |
| ------------------------- | ------------------- |
| Ramp rate                 | 5, 10, 15 °C/min    |
| Carburization temperature | 550–900 °C          |
| Hold time                 | 0.5, 2, 5, 10 hr    |
| Gas                       | H₂ or N₂            |
| Flow rate                 | 30, 60, 100 sccm    |
| AMT : sucrose ratio       | 1:0.5, 1:1, 1:2 g/g |

Fixed conditions: 1 g ammonium heptamolybdate tetrahydrate, 120 °C drying for 24 h, 212 µm sieve, 0.6 g loaded sample, N₂ cooling, and 1% O₂/N₂ passivation.

---

## Repository Structure

```text
boicl_crystal_phase_isolation/
│
├── README.md
├── LICENSE
├── bo_icl_env.yml
├── cubic_WC_boicl.ipynb                  # Main BOICL campaign notebook
├── archive.png                           # Repository/project figure
│
├── boicl/                                # Core BOICL package
│   ├── __init__.py
│   ├── aqfxns.py                         # Acquisition functions: EI, PI, UCB, greedy
│   ├── asktell.py                        # Main Ask/Tell BOICL interface
│   ├── asktellFinetuning.py              # Fine-tuning-oriented Ask/Tell variant
│   ├── asktellGPR.py                     # Gaussian process regression variant
│   ├── asktellNearestNeighbor.py         # Nearest-neighbor baseline variant
│   ├── asktellRidgeRegression.py         # Ridge regression baseline variant
│   ├── call_design_space.py              # Design-space helper script
│   ├── design_space.py                   # Design-space utilities
│   ├── llm_model.py                      # LLM API wrapper
│   ├── pool.py                           # Candidate pool, embeddings, FAISS indexing
│   ├── tool.py                           # LangChain tool integration
│   └── version.py
│
├── BOICL_docs/                           # Prompt templates
│   ├── pred_system_message.txt           # Forward-model system prompt
│   ├── inv_system_message.txt            # Inverse-design system prompt
│   └── inv_system_message_syn_data.txt   # Synthetic-data inverse-design prompt
│
├── dataset/
│   ├── Metal_Sucrose_DesignSpace_fr_T1.xlsx
│   └── synthetic_Mo-Carburization_Dataset_v2_labeled.xlsx
│
├── campaigns/
│   ├── Synthetic/                        # Synthetic benchmark outputs
│   │   ├── boicl_mo2c_alpha_live_20260520_v3_mo_carburization_dataset_v2_unlabeled_20260520_155838_observations.csv
│   │   ├── boicl_mo2c_alpha_offline_20260518_mo_carburization_dataset_v2_labeled_20260520_122909_observations.csv
│   │   └── synthetic_data.png
│   │
│   └── Trajectory_1_with_seeds_EI_acq/   # Experimental T1 XRD files
│       ├── M7-Mo2C-1:1_15C_min_800C_.5hr_60sccm_N2.csv
│       ├── M12-Mo2C-1:1_15C_min_600C_2hr_30sccm_H2.csv
│       ├── M14-Mo2C-1:1_15C_min_900C_.5hr_30sccm_H2.csv
│       ├── M16-Mo2C-1:1_10C_min_620C_10hr_60sccm_N2.dat
│       ├── M17-Mo2C-1:1_15C_min_620C_.5hr_30sccm_H2.dat
│       ├── M21-Mo2C-1:1_10C_min_620C_.5hr_60sccm_N2.dat
│       ├── M23-Mo2C-1:1_10C_min_630C_2hr_60sccm_H2.dat
│       └── M24-Mo2C-1:2_10C_min_690C_.5hr_30sccm_N2.dat
│
├── xrd/
│   ├── smartlab_Si640g.instprm.bak       # Rigaku SmartLab instrument parameters
│   └── Reference cif/                    # CIF files for Rietveld refinement
│       ├── MoC_cubic_fixed.cif           # Target cubic MoC phase
│       ├── MoC_0.66.cif
│       ├── MoC_0.68.cif
│       ├── MoC_0.70.cif
│       ├── MoC_0.72.cif
│       ├── MoC_0.74.cif
│       ├── MoC_0.75.cif
│       ├── Mo2C_Pbcn_fixed.cif
│       ├── Mo_Im3m_fixed.cif
│       ├── MoO2_P21c_fixed.cif
│       ├── graphite_2H_P63mmc.cif
│       ├── W_Im-3m_mp-91.cif             # Auxiliary/legacy W reference
│       └── W_Pm-3n_mp-11334.cif          # Auxiliary/legacy W reference
│
└── archive/                              # Analysis artifacts, plots, and earlier outputs
    ├── chemistry_embeddings.ipynb         # Embedding model comparison
    ├── alpha_moc_bo_inverse_design.pdf
    ├── alpha_moc_bo_inverse_design.png
    ├── boicl_trajectory.pdf
    ├── boicl_trajectory.png
    ├── campaigns_three_panel.pdf
    ├── campaigns_three_panel.png
    ├── campaigns_two_panel.pdf
    ├── campaigns_two_panel.png
    ├── live_moc_trajectory.png
    ├── rietveld_best_T1.pdf
    ├── rietveld_best_T1.png
    ├── rietveld_best_eva.pdf
    ├── rietveld_best_eva.png
    ├── rietveld_M7.png
    ├── rietveld_M26.png
    ├── xrd_best_T1.xlsx
    ├── xrd_summary1.xlsx
    ├── best_best_xrd_summary.xlsx
    ├── best_human guided xrd.xlsx
    ├── gp_bo_trajectories.csv
    ├── gp_bo_all_replicates.npy
    ├── matbert_eval_texts.json
    └── out/
        ├── mo_carb_gpt-4o_2_1_1_review1_comment2_v1.pkl
        └── mo_carb_gpt-4o_bert_1_1_1_v1.pkl
```

Local-only files such as `.env`, `.emb_cache/`, `.DS_Store`, `__pycache__/`, `repo_tree.txt`, and `tracked_files.txt` should not be committed.

---

## How It Works

### Forward Model

GPT-4o is prompted as a materials scientist specializing in Mo-carbide synthesis. Given a synthesis procedure string, it outputs a predicted cubic MoC weight fraction from 0–100%.

Few-shot examples are selected from the labeled pool using semantic similarity. Procedure strings are embedded and indexed with FAISS, allowing the model to retrieve chemically relevant examples for each new query.

### Inverse Model

A second GPT-4o call can be used for inverse design. Given a target MoC weight fraction, the model proposes a synthesis procedure that may be added to the candidate pool.

### Acquisition

At each iteration, the model predicts cubic MoC weight fraction for unlabeled candidates in the pool. The T1 campaign uses Expected Improvement:

* **EI** selects the candidate maximizing expected improvement over the current best observed result

The selected procedure is synthesized, characterized by XRD, and added back to the labeled set using the Ask/Tell loop.

### Pool Management

Candidate procedures are embedded and indexed with FAISS. The pool utilities support approximate sampling and few-shot selection using similarity and diversity-aware retrieval.

---

## Setup

### Environment

Create and activate the conda environment:

```bash
conda env create -f bo_icl_env.yml
conda activate bo_icl
```

### API Keys

Create a local `.env` file in the repository root:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit `.env` or any file containing API keys.

The main campaign uses GPT-4o for prediction and `text-embedding-3-large` for embeddings through the OpenAI API.

### Optional Pre-built Pool Cache

To skip re-embedding the full design space on first use, place the following cache files in the repository root:

```text
mo2c_sucrose_prompts.pkl
mo2c_sucrose_emb.npy
mo2c_sucrose_index.faiss
```

If the embedding/index files are absent, the pool can be regenerated and cached locally.

---

## Running the Campaign

Open the main notebook:

```text
cubic_WC_boicl.ipynb
```

Run the notebook sections in order for the T1 Expected Improvement campaign.

General workflow:

1. Load the T1 design space from `dataset/Metal_Sucrose_DesignSpace_fr_T1.xlsx`
2. Initialize the BOICL Ask/Tell loop
3. Update completed experimental labels with newly measured cubic MoC weight fractions
4. Run prediction and acquisition to select the next suggested synthesis
5. Synthesize the selected candidate
6. Characterize by powder XRD and Rietveld refinement
7. Feed the measured cubic MoC result back into the campaign

Raw experimental XRD files for the T1 trajectory are stored in:

```text
campaigns/Trajectory_1_with_seeds_EI_acq/
```

---

## Phase Characterization

Raw powder XRD patterns in `campaigns/` are refined using GSAS-II with the reference CIF files in `xrd/Reference cif/`.

The Rietveld refinement reports weight fractions for phases including:

* **α-MoC (`Fm3̄m`)** — target cubic phase
* **β-Mo₂C (`Pbcn`)** — carbide byproduct
* **Mo (`Im3̄m`)** — unreacted metal
* **MoO₂ (`P2₁/c`)** — oxide byproduct
* **Graphite** — excess carbon phase

The cubic MoC weight fraction is the optimization target.

---

## Embedding Model Comparison

`archive/chemistry_embeddings.ipynb` benchmarks embedding models for Mo-carburization synthesis text. The comparison evaluates whether embedding distance captures chemically meaningful changes in synthesis conditions rather than superficial text changes.

Summary from the current analysis:

| Model                    | SNR (chem/surface) | Unit sensitivity | Gas ranking |
| ------------------------ | -----------------: | ---------------: | ----------: |
| `text-embedding-ada-002` |           **3.40** |             0.40 |        1/10 |
| `matscibert`             |               2.31 |             0.56 |        5/10 |
| `text-embedding-3-large` |               1.48 |             0.79 |        5/10 |

The results indicate that general-purpose text embeddings can capture some synthesis-text similarity, but numerical parameter sensitivity remains limited.

---

## Dependencies

Key packages are listed in `bo_icl_env.yml`.

Major dependencies include:

| Package                | Purpose                              |
| ---------------------- | ------------------------------------ |
| `openai`               | LLM API and embeddings               |
| `anthropic`            | Optional Claude API support          |
| `faiss-cpu`            | Vector similarity search             |
| `langchain`            | Prompt chaining and tool integration |
| `botorch` / `gpytorch` | Gaussian-process BO baseline         |
| `transformers`         | HuggingFace / MatBERT workflows      |
| `scikit-learn`         | Similarity metrics and preprocessing |
| `numpy` / `scipy`      | Numerical computing                  |
| `pandas` / `openpyxl`  | Dataset and Excel I/O                |

Install exact pinned versions using the conda environment file.

---

## Notes on Repository Hygiene

The following files should remain local and should not be committed:

```text
.env
.env.*
.DS_Store
__pycache__/
*.pyc
repo_tree.txt
tracked_files.txt
.emb_cache/
```

If these files appear in `git status`, remove them from tracking or delete the generated helper files before committing.

---

## Citation

If you use this code or dataset, please cite the associated work.

Citation details forthcoming.
