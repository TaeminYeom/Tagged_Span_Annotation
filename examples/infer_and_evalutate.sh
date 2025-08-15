MODEL=o3
METHOD=TSA

SRC_LANG=en
TGT_LANGS=("cs" "ja" "zh" "is" "uk" "ru")

INPUT_DIR=data

OUTPUT_DIR=output/$METHOD/$MODEL
mkdir -p $OUTPUT_DIR

RESULT_DIR=result/$METHOD/$MODEL
mkdir -p $RESULT_DIR

LOG_DIR=log/$METHOD/$MODEL
mkdir -p $LOG_DIR

for TGT_LANG in "${TGT_LANGS[@]}"; do
    echo "Processing language pair: ${SRC_LANG}-${TGT_LANG}"
    
    # Check if inference output already exists
    JSONL_OUTPUT=$OUTPUT_DIR/mteval-task2-dev.${SRC_LANG}-${TGT_LANG}.jsonl
    if [ -f "$JSONL_OUTPUT" ]; then
        echo "Inference output already exists, skipping inference for ${SRC_LANG}-${TGT_LANG}"
    else
        echo "Running inference for ${SRC_LANG}-${TGT_LANG}"
        stdbuf -oL -eL python -u inference/$METHOD.py \
            --input_tsv $INPUT_DIR/mteval-task2-dev.${SRC_LANG}-${TGT_LANG}.tsv \
            --endpoint $ENDPOINT \
            --api_key $KEY \
            --api_version preview \
            --deployment_name $MODEL \
            --output $JSONL_OUTPUT \
            --source_lang $SRC_LANG \
            --target_lang $TGT_LANG \
            --worker 150 | tee $LOG_DIR/${SRC_LANG}-${TGT_LANG}.log
    fi

    python scripts/convert_jsonl_to_tsv.py \
        --input $JSONL_OUTPUT \
        --output $OUTPUT_DIR/mteval-task2-dev.${SRC_LANG}-${TGT_LANG}.tsv

    python scripts/evaluate.py \
        --references_tsv $INPUT_DIR/mteval-task2-dev.${SRC_LANG}-${TGT_LANG}.tsv \
        --predictions_tsv $OUTPUT_DIR/mteval-task2-dev.${SRC_LANG}-${TGT_LANG}.tsv \
        > $RESULT_DIR/mteval-task2-dev.${SRC_LANG}-${TGT_LANG}.eval

    echo "Completed processing for ${SRC_LANG}-${TGT_LANG}"
done
