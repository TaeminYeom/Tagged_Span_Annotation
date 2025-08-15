# Tagged Span Annotation (TSA)

This repository contains reference code to try the **Tagged Span Annotation (TSA)** method for span-level MT error detection and to compare it against alternative prompting styles.

## 1. Dataset Preparation

```bash
# Prepare the WMT24 ESA annotation dataset for evaluation
bash scripts/prepare_dataset.sh
```

---

## 2. Quick Start

```bash
export ENDPOINT=<your OpenAI endpoint>
export KEY=<your OpenAI API key>

bash examples/infer_and_evaluate.sh
```

The script runs inference and then evaluates predictions on the WMT24 ESA annotations.

---

## 3. Inference Scripts

| File | Purpose | Prompting / Output style |
|------|---------|--------------------------|
| `TSA.py`            | **Tagged Span Annotation** main implementation | Tagged inline spans |
| `GEMBA.py`          | Ablation: error spans returned as **text**. Same prompt as 'TSA.py'     | text spans |
| `Direct-index.py`   | Ablation: error spans as **character indices**. Same prompt as 'TSA.py' | index spans |
| `TSA_no_prec.py`    | Ablation: TSA **without** the precision-emphasis prompt | Tagged inline spans |
| `GEMBA_original.py` | Baseline: Original **GEMBA-MQM** prompt (only output is forced to JSON) | text spans |
