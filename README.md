# Deep-learning electrocardiographic age varies widely between repeat recordings and dilutes associations with mortality

Code for the study of the same title: the retest reliability of age estimated from a 12-lead electrocardiogram (ECG) by ten deep-learning models
in repeat ECGs of the same patients in MIMIC-IV-ECG, the sources of its variation, its consequences for the association of the age gap with
death within 1 year, and every table and figure of the paper.

> Authors, citation and licence: to be added on publication.

## What the study does

The age gap, AI-ECG age minus chronological age, has been proposed as a marker of biological aging. The study measures how much AI-ECG age
varies between two ECGs of the same patient recorded within 1 hour, within 48 hours and within 30 days in outpatient care; how much of this
variation is explained by the time of day, the care setting and the physiology of the patient; and how the variation weakens the association
between the age gap and death within 1 year and changes individual flags of an age gap of 8 years or more.

The ten models are the public model of Lima et al. and three ECG foundation models (ECGFounder, ECG-FM, HuBERT-ECG), each used in three ways:
frozen, with ridge regression on its embedding trained on MIMIC-IV-ECG; fine-tuned on MIMIC-IV-ECG; and fine-tuned on CODE-15%.

| Analysis | What it measures | Notebook |
|---|---|---|
| Accuracy | mean absolute error of AI-ECG age in MIMIC-IV-ECG, CODE-15% and PTB-XL | 04 |
| Retest reliability | standard error of measurement, smallest detectable change, intraclass correlation of the age gap | 05 |
| Sources of variation | Shapley decomposition over time of day, care setting and physiology; night–daytime comparison | 05 |
| Pull toward the mean age | slope of AI-ECG age on chronological age | 05 |
| Mortality | odds ratio for death within 1 year per 10-year larger age gap; SIMEX; averaging of repeat ECGs; changes of an 'older' flag | 06 |

## Contents

Everything is in `notebooks/`, to be run in this order. There are no Python modules: settings and shared functions live in notebook 00,
which every other notebook loads with `%run ./00_setup_and_settings.ipynb`.

| Notebook | Content | Needs |
|---|---|---|
| `00_setup_and_settings.ipynb` | environment check, **global settings** (the master folder and every path), constants, the ten models, shared functions | – |
| `01_data_download_and_preprocessing.ipynb` | where to obtain each dataset and under which conditions; download; waveform arrays, machine measurements, quality control, report statements, HuBERT-ECG inputs | internet; disk space (below) |
| `02_linkage_cohort_and_sets.ipynb` | care setting, age, sex and death within 1 year of each ECG; exclusions; patient-level splits; repeat-ECG sets; mortality and averaging cohorts | 01; credentialed access |
| `03_models_and_training.ipynb` | pretrained weights and the developers' code; frozen embeddings with ridge and logistic regression; fine-tuning on MIMIC-IV-ECG and on CODE-15% | 01, 02; one NVIDIA graphics card |
| `04_predictions_and_accuracy.ipynb` | AI-ECG age of the test-split ECGs with the ten models; amplitude scale of the Lima et al. model; accuracy | 03; one NVIDIA graphics card |
| `05_reliability_analysis.ipynb` | retest reliability in every repeat-ECG set, distributions of the differences, sources of variation, night–daytime comparison, predicted sex, pull toward the mean age | 02, 04 |
| `06_mortality_analysis.ipynb` | association of the age gap with death within 1 year, SIMEX, averaging of repeat ECGs, 'older' flags | 02, 04 |
| `07_tables.ipynb` | Tables 1–2 and Supplementary Tables 1–4 | 02–06 |
| `08_figures.ipynb` | Figs. 1–4 and Supplementary Figs. 1–6 | 02, 04–06 |

In notebooks 07 and 08 every table and figure has its own heading (`## Table 2`, `## Supplementary Fig. 1`) with the caption of the paper above
the code that draws it.

## Getting started

1. **Environment** (Python 3.12; the versions used for the paper are pinned):

   ```bash
   conda env create -f environment.yml
   conda activate ai-ecg-age-repeatability
   python -m ipykernel install --user --name ai-ecg-age-repeatability
   jupyter lab
   ```

   or `pip install -r requirements.txt` in a Python 3.12 environment. Both install the PyTorch build for CUDA 12.8; the files say how to
   install the build without a graphics card.

2. **Master folder.** Choose a folder with enough free space for the data, the models and the results, and either set the environment
   variable `STUDY_MASTER` to it or edit the line `MASTER = ...` in the cell *Global settings* of `notebooks/00_setup_and_settings.ipynb`.
   Nothing is read or written outside this folder.

3. **Data and pretrained models** are not part of this repository. Notebook 01 explains for each dataset where it comes from, what access it
   requires and where the files go; notebook 03 does the same for the pretrained weights and for the developers' code, which is cloned into
   `third_party/` inside the master folder.

4. **Run the notebooks in order.** Long steps skip outputs that exist, so they can be interrupted and resumed. For a short functional check,
   notebook 01 has the switch `SUBSET` (a few records of each dataset) and notebooks 03 and 04 the switch `QUICK_TEST`;
   `GPU_MEMORY_FRACTION` in notebook 00 limits the share of the memory of the graphics card that a notebook may use.

## Data

| Dataset | Used for | Source | Access |
|---|---|---|---|
| MIMIC-IV-ECG 1.0 | ECGs of the training, validation and test splits | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) | open (Open Data Commons Open Database License 1.0) |
| MIMIC-IV 3.1 | care setting, age, sex, death within 1 year | [PhysioNet](https://physionet.org/content/mimiciv/3.1/) | credentialed: PhysioNet account, training, data use agreement |
| MIMIC-IV-ED 2.2 | emergency department stays | [PhysioNet](https://physionet.org/content/mimic-iv-ed/2.2/) | credentialed, as MIMIC-IV |
| CODE-15% 1.0.0 | fine-tuning on CODE-15%; accuracy in its test split | [Zenodo](https://zenodo.org/records/4916206) | open (CC BY 4.0) |
| PTB-XL 1.0.3 | accuracy; amplitude scale of the Lima et al. model | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) | open (CC BY 4.0) |

Read the terms of each dataset on its own page before using it. Everything derived from MIMIC-IV and MIMIC-IV-ED stays inside the master folder,
the notebooks print aggregate numbers only, and counts between 1 and 9 are shown as "<10". Under the data use agreement, derived data,
predictions and models trained with MIMIC-IV labels may not be shared.

Disk space: about 170 GB of raw data (plus about 85 GB of ZIP files during the download, which can be deleted afterwards), about 170 GB of
waveform arrays and model inputs, and about 5 GB of embeddings.

## Pretrained models

| Model | Weights | Code | Licence stated by its developers |
|---|---|---|---|
| Lima et al. | [Zenodo 4892365](https://doi.org/10.5281/zenodo.4892365) | [antonior92/ecg-age-prediction](https://github.com/antonior92/ecg-age-prediction) | weights CC BY 4.0, code MIT |
| ECGFounder | [PKUDigitalHealth/ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder) | [PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) | MIT (model card) |
| ECG-FM | [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) | [bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM), [Jwoo5/fairseq-signals](https://github.com/Jwoo5/fairseq-signals) | MIT |
| HuBERT-ECG large | [Edoardo-Coppola/hubert-ecg-large](https://huggingface.co/Edoardo-Coppola/hubert-ecg-large) | [Edoar-do/HuBERT-ECG](https://github.com/Edoar-do/HuBERT-ECG) | CC BY-NC 4.0 (non-commercial use only) |

Notebook 03 gives the revisions and checksums. No code or weights of these projects are copied into this repository. The encoder of ECG-FM
is written out in plain PyTorch in notebook 00, with the parameter names of the released checkpoint; HuBERT-ECG uses the HuBERT network of
`transformers`.

## Notes

- Results that involve random sampling (confidence intervals, SIMEX) or computations on the graphics card can differ slightly from those
  reported in the paper.
- Hardware: one NVIDIA GeForce RTX 5090 (32 GB) and a 24-core CPU. Approximate run times: notebook 01 about 3 hours (reading MIMIC-IV-ECG about
  1 hour, CODE-15% about 45 minutes, HuBERT-ECG inputs about 45 minutes); 02 about 1 minute; 03 about 7 hours (embeddings about 10 minutes per
  foundation model, ridge and logistic regression 1–2 minutes, each fine-tuning 20–110 minutes); 04 about 1 hour; 05 about 30 minutes and 06
  about 2 hours with four CPU threads; 07 and 08 under 1 minute. Only notebooks 03 and 04 use the graphics card.
- The notebooks are stored without outputs.

## Repository layout

```
├── README.md
├── environment.yml        conda environment with the versions used
├── requirements.txt       the same versions for pip
└── notebooks/             00 … 08, see above
```
