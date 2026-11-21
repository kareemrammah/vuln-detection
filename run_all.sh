#!/bin/bash
# Reproduces every experiment in the report. Run from the repository root.
# Time on one Colab T4: CodeBERT ~50 min/run, BiLSTM ~11 min/run. Stop whenever you run out of compute.
set -e; cd "$(dirname "$0")"; export PYTHONPATH=src
python src/eda.py                              # data statistics + report/eda.png
python src/length_baseline.py                  # length-only reference
for s in 0 1 2; do                             # main comparison, 3 seeds
  python src/train.py --model bilstm   --seed $s
  python src/train.py --model codebert --seed $s
done
# learning-rate tuning (seed 0, chosen by best validation F1)
python src/train.py --model bilstm   --lr 5e-4 --seed 0
python src/train.py --model bilstm   --lr 2e-3 --seed 0
python src/train.py --model codebert --lr 1e-5 --seed 0
for s in 0 1 2; do python src/train.py --model codebert --lr 3e-5 --seed $s; done
# ablations (seed 0)
python src/train.py --model bilstm_pt --seed 0
python src/train.py --model bilstm   --strategy headtail --seed 0
python src/train.py --model codebert --strategy headtail --seed 0
for L in 128 256; do python src/train.py --model codebert --max_len $L --seed 0; done
python src/train.py --model codebert --train_top_k 4 --seed 0
python src/analyze.py                          # tables, curves, length analysis, confident errors
python src/brace_artifact.py                   # formatting-artifact check
