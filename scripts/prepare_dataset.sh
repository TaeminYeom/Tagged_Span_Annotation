#!/bin/bash

set -e

if ! command -v pipx &> /dev/null; then
    pip install pipx --user
    pipx ensurepath
fi

if ! command -v uv &> /dev/null; then
    pipx install uv
fi

# Create a tsv format file for wmt24_esa file from wmt24_esa.jsonl
WMT24_ESA_JSONL=wmt24_esa.jsonl
if [ ! -f $WMT24_ESA_JSONL ]; then
    wget https://github.com/wmt-conference/wmt24-news-systems/raw/refs/heads/main/jsonl/wmt24_esa.jsonl
fi

WMT24_DATA_DIR=data
DEV_TSV=$WMT24_DATA_DIR/mteval-task2-dev.tsv
mkdir -p $WMT24_DATA_DIR

uv run --python 3.11 \
    --with pandas \
    --with datasets \
    scripts/create_tsv_from_wmt24_esa.py \
    --wmt24_esa_jsonl $WMT24_ESA_JSONL \
    --output_tsv $DEV_TSV \
    --filter_data_with_invalid_span

LANGS=("en-cs" "en-ja" "en-zh" "en-is" "en-uk" "en-ru")
for LANG in "${LANGS[@]}"; do
    uv run --python 3.11 \
    --with pandas \
    scripts/extract_from_tsv.py \
        --input_tsv $DEV_TSV \
        --langs $LANG \
        --output_tsv $WMT24_DATA_DIR/mteval-task2-dev.$LANG.tsv
done
