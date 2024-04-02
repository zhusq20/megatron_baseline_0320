#! /bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="logs/log_${timestamp}.txt"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=8848
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_PATH=/mnt/octave/data/dataset/llama/llama-2-7b-chat-hf
TOKENIZER_MODEL=/mnt/octave/data/dataset/llama/llama-2-7b-chat-hf
DATA_PATH=/mnt/octave/data/siqizhu/megatron0.4.0/data-spm/redarxiv_test_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_ARGS="
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
"

seq_lens=(1024)
max_batch=( 4 )
#tp=(2 1 4 1 2 1)
#pp=(2 4 1 2 1 1)
#dp=(2 2 2 4 4 8)
tp=(1)
pp=(8)
dp=(1)
num_micro_batch=( 128 )
for i in "${!seq_lens[@]}"; do
    seq_len=${seq_lens[$i]}
    max_batch=${max_batch[$i]}
    for batch in $(seq 1 1 $max_batch);
    do
            for k in "${!tp[@]}"; do
                t=${tp[$k]}
                p=${pp[$k]}
                d=${dp[$k]}
                for m in "${!num_micro_batch[@]}"; do
                    num_micro_batch=${num_micro_batch[$m]}
                    global_bs=$((num_micro_batch * batch))
                    LOG_PATH=/mnt/octave/data/siqizhu/useless/tensorboard-log-Jan5/Llama-2-70b_seq-len${seq_len}_global_batch${global_bs}_gpu${WORLD_SIZE}_tp${t}pp${p}
                    GPT_ARGS="
                        --tensor-model-parallel-size $t \
                        --pipeline-model-parallel-size $p \
                        --micro-batch-size $batch \
                        --global-batch-size $global_bs \
                        --load-model-from-hf-config \
                        --model-name-or-path $MODEL_PATH \
                        --seq-length $seq_len \
                        --decoder-seq-length $seq_len \
                        --max-position-embeddings $seq_len \
                        --lr 0.00015 \
                        --train-iters 5 \
                        --lr-decay-iters 320000 \
                        --lr-decay-style cosine \
                        --min-lr 1.0e-5 \
                        --weight-decay 1e-2 \
                        --lr-warmup-fraction .01 \
                        --clip-grad 1.0 \
                        --position-embedding-type rope \
                        --untie-embeddings-and-output-weights \
                        --disable-bias-linear \
                        --fp16 \
			--use-flash-attn \
                    "
                    OUTPUT_ARGS="
                        --log-interval 1 \
                        --save-interval 10000 \
                        --eval-interval 1000 \
                        --eval-iters 0 \
                        --log-timers-to-tensorboard \
                        --log-world-size-to-tensorboard \
                        --log-memory-to-tensorboard \
                        --tensorboard-dir $LOG_PATH \
                        --timing-log-level 1 \
                        --tensorboard-log-interval 1 \
                    "
                    echo "[seq_len $seq_len, micro_batch $batch, tp $t, pp $p, dp $d, num_micro_batch $num_micro_batch]"
                    torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
                    $GPT_ARGS \
                    $DATA_ARGS \
                    $OUTPUT_ARGS \
                    --distributed-backend nccl 

		    echo Current date is $(date) 
                done
            done
            # echo "[seq_len $seq_len, batch $batch]"
            # torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
            # $GPT_ARGS \
            # $DATA_ARGS \
            # $OUTPUT_ARGS \
            # --distributed-backend nccl
    done
done



# torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
#     $GPT_ARGS \
#     $DATA_ARGS \
#     $OUTPUT_ARGS \
#     --distributed-backend nccl
