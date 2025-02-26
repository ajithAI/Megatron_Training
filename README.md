# Megatron_Training

# Megatron_Traininig

#### Pull Docker container 
```
docker run -it --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH --security-opt seccomp=unconfined \ 
    -v /mnt/Scratch_space/ajith:/home/user -w /home/user --name Megatron_Training_8xH100 nvcr.io/nvidia/pytorch:25.02-py3 bash
```
```
git clone https://github.com/NVIDIA/Megatron-LM
```
```
from datasets import load_dataset
train_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
train_data.to_json("wikitext_data.json", lines=True)
```
```
pip3 install nltk
```
```
wget https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt -O merges.txt
wget https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json -O vocab.json
```

```
python3 tools/preprocess_data.py --input wikitext_data.json --output-prefix wikitext --vocab-file vocab.json  --tokenizer-type GPT2BPETokenizer  --merge-file merges.txt --workers 32 --append-eod
```

```
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch  $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        --seq-length 1024 --bf16 --micro-batch-size 1 --global-batch-size 8 \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH_New \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH
```
```
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
pretrain_gpt.py \
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 1 \
--micro-batch-size 1 \
--vocab-file vocab.json \
--merge-file merges.txt \
--save /home/user/Llama-3.1-8B-Instruct_Ajith \
--load /home/user/Llama-3.1-8B-Instruct \
--data-path wikitext_text_document \
--seq-length 1024 \
--max-position-embeddings 131072 \
--tokenizer-type GPT2BPETokenizer \
--tokenizer-model /home/user/Llama-3.1-8B-Instruct \
--load /home/user/Llama-3.1-8B-Instruct  \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--disable-bias-linear \
--transformer-impl transformer_engine \
--group-query-attention 8 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--rotary-base 500000 \
--rotary-percent 1.0 \
--use-rope-scaling \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--swiglu \
--bf16 \
```

```
usage: pretrain_gpt.py [-h] [--num-layers NUM_LAYERS] [--encoder-num-layers ENCODER_NUM_LAYERS] [--decoder-num-layers DECODER_NUM_LAYERS] [--hidden-size HIDDEN_SIZE] [--ffn-hidden-size FFN_HIDDEN_SIZE]
                       [--num-attention-heads NUM_ATTENTION_HEADS] [--attention-backend {AttnBackend.flash,AttnBackend.fused,AttnBackend.unfused,AttnBackend.local,AttnBackend.auto}] [--kv-channels KV_CHANNELS]
                       [--group-query-attention] [--num-query-groups NUM_QUERY_GROUPS] [--max-position-embeddings MAX_POSITION_EMBEDDINGS] [--position-embedding-type {learned_absolute,rope,relative,none}]
                       [--relative-attention-num-buckets RELATIVE_ATTENTION_NUM_BUCKETS] [--relative-attention-max-distance RELATIVE_ATTENTION_MAX_DISTANCE] [--use-rotary-position-embeddings] [--rotary-base ROTARY_BASE]
                       [--rotary-percent ROTARY_PERCENT] [--rotary-interleaved] [--rotary-seq-len-interpolation-factor ROTARY_SEQ_LEN_INTERPOLATION_FACTOR] [--use-rope-scaling] [--rope-scaling-factor ROPE_SCALING_FACTOR]
                       [--no-position-embedding] [--make-vocab-size-divisible-by MAKE_VOCAB_SIZE_DIVISIBLE_BY] [--normalization {LayerNorm,RMSNorm}] [--norm-epsilon NORM_EPSILON] [--apply-layernorm-1p]
                       [--apply-residual-connection-post-layernorm] [--openai-gelu] [--squared-relu] [--swiglu] [--onnx-safe ONNX_SAFE] [--bert-no-binary-head] [--untie-embeddings-and-output-weights] [--multi-latent-attention]
                       [--attention-dropout ATTENTION_DROPOUT] [--hidden-dropout HIDDEN_DROPOUT] [--weight-decay WEIGHT_DECAY] [--start-weight-decay START_WEIGHT_DECAY] [--end-weight-decay END_WEIGHT_DECAY]
                       [--weight-decay-incr-style {constant,linear,cosine}] [--clip-grad CLIP_GRAD] [--adam-beta1 ADAM_BETA1] [--adam-beta2 ADAM_BETA2] [--adam-eps ADAM_EPS] [--sgd-momentum SGD_MOMENTUM]
                       [--micro-batch-size MICRO_BATCH_SIZE] [--batch-size BATCH_SIZE] [--global-batch-size GLOBAL_BATCH_SIZE] [--rampup-batch-size [RAMPUP_BATCH_SIZE ...]] [--decrease-batch-size-if-needed] [--recompute-activations]
                       [--recompute-granularity {full,selective}] [--no-check-for-nan-in-loss-and-grad] [--check-for-spiky-loss] [--check-for-large-grads] [--distribute-saved-activations] [--recompute-method {uniform,block}]
                       [--recompute-num-layers RECOMPUTE_NUM_LAYERS] [--no-clone-scatter-output-in-embedding] [--profile] [--profile-step-start PROFILE_STEP_START] [--profile-step-end PROFILE_STEP_END]
                       [--iterations-to-skip ITERATIONS_TO_SKIP [ITERATIONS_TO_SKIP ...]] [--result-rejected-tracker-filename RESULT_REJECTED_TRACKER_FILENAME] [--disable-gloo-process-groups] [--use-pytorch-profiler]
                       [--profile-ranks PROFILE_RANKS [PROFILE_RANKS ...]] [--record-memory-history] [--memory-snapshot-path MEMORY_SNAPSHOT_PATH] [--tp-comm-overlap] [--tp-comm-overlap-cfg TP_COMM_OVERLAP_CFG]
                       [--disable-tp-comm-overlap-ag] [--disable-tp-comm-overlap-rs] [--tp-comm-overlap-rs-dgrad] [--disable-tp-comm-bulk-dgrad] [--disable-tp-comm-bulk-wgrad] [--tp-comm-bootstrap-backend {nccl,mpi,gloo}]
                       [--use-cpu-initialization] [--empty-unused-memory-level {0,1,2}] [--deterministic-mode] [--check-weight-hash-across-dp-replicas-interval CHECK_WEIGHT_HASH_ACROSS_DP_REPLICAS_INTERVAL]
                       [--calculate-per-token-loss] [--train-sync-interval TRAIN_SYNC_INTERVAL] [--checkpoint-activations] [--train-iters TRAIN_ITERS] [--train-samples TRAIN_SAMPLES] [--log-interval LOG_INTERVAL]
                       [--exit-interval EXIT_INTERVAL] [--exit-duration-in-mins EXIT_DURATION_IN_MINS] [--exit-signal-handler] [--tensorboard-dir TENSORBOARD_DIR] [--no-masked-softmax-fusion] [--no-bias-gelu-fusion]
                       [--no-bias-swiglu-fusion] [--no-bias-dropout-fusion] [--no-rope-fusion] [--cross-entropy-loss-fusion] [--use-flash-attn] [--disable-bias-linear] [--add-qkv-bias] [--optimizer {adam,sgd}]
                       [--optimizer-cpu-offload] [--optimizer-offload-fraction OPTIMIZER_OFFLOAD_FRACTION] [--use-torch-optimizer-for-cpu-offload] [--overlap-cpu-optimizer-d2h-h2d] [--no-pin-cpu-grads] [--no-pin-cpu-params]
                       [--dataloader-type {single,cyclic,external}] [--no-async-tensor-model-parallel-allreduce] [--no-persist-layer-norm] [--sequence-parallel] [--no-gradient-accumulation-fusion] [--use-mcore-models]
                       [--use-legacy-models] [--manual-gc] [--manual-gc-interval MANUAL_GC_INTERVAL] [--no-manual-gc-eval] [--disable-tp-comm-split-ag] [--disable-tp-comm-split-rs] [--pipeline-model-parallel-comm-backend {nccl,ucc}]
                       [--seed SEED] [--data-parallel-random-init] [--init-method-std INIT_METHOD_STD] [--init-method-xavier-uniform] [--lr LR] [--lr-decay-style {constant,linear,cosine,inverse-square-root,WSD}]
                       [--lr-wsd-decay-style {exponential,linear,cosine}] [--lr-decay-iters LR_DECAY_ITERS] [--lr-decay-samples LR_DECAY_SAMPLES] [--lr-wsd-decay-samples LR_WSD_DECAY_SAMPLES] [--lr-wsd-decay-iters LR_WSD_DECAY_ITERS]
                       [--lr-warmup-fraction LR_WARMUP_FRACTION] [--lr-warmup-iters LR_WARMUP_ITERS] [--lr-warmup-samples LR_WARMUP_SAMPLES] [--lr-warmup-init LR_WARMUP_INIT] [--warmup WARMUP] [--min-lr MIN_LR]
                       [--override-opt_param-scheduler] [--use-checkpoint-opt_param-scheduler] [--decoupled-lr DECOUPLED_LR] [--decoupled-min-lr DECOUPLED_MIN_LR] [--save SAVE] [--save-interval SAVE_INTERVAL] [--no-save-optim]
                       [--no-save-rng] [--load LOAD] [--no-load-optim] [--no-load-rng] [--non-persistent-save-interval NON_PERSISTENT_SAVE_INTERVAL] [--non-persistent-ckpt-type {global,local,in_memory,None}]
                       [--non-persistent-global-ckpt-dir NON_PERSISTENT_GLOBAL_CKPT_DIR] [--non-persistent-local-ckpt-dir NON_PERSISTENT_LOCAL_CKPT_DIR] [--non-persistent-local-ckpt-algo {fully_parallel,atomic}] [--finetune]
                       [--pretrained-checkpoint PRETRAINED_CHECKPOINT] [--ckpt-step CKPT_STEP] [--no-initialization] [--use-checkpoint-args] [--use-mp-args-from-checkpoint-args] [--no-use-tokenizer-model-from-checkpoint-args]
                       [--exit-on-missing-checkpoint] [--use-dist-ckpt] [--use-persistent-ckpt-worker] [--auto-detect-ckpt-format] [--dist-ckpt-format DIST_CKPT_FORMAT_DEPRECATED] [--ckpt-format {torch,torch_dist,zarr}]
                       [--ckpt-convert-format {torch,torch_dist,zarr}] [--ckpt-convert-save CKPT_CONVERT_SAVE] [--ckpt-convert-update-legacy-dist-opt-format] [--ckpt-fully-parallel-save] [--no-ckpt-fully-parallel-save] [--async-save]
                       [--ckpt-fully-parallel-load] [--ckpt-assume-constant-structure] [--dist-ckpt-strictness {assume_ok_unexpected,log_unexpected,log_all,raise_unexpected,raise_all,return_unexpected,return_all,ignore_all}] [--fp16]
                       [--bf16] [--loss-scale LOSS_SCALE] [--initial-loss-scale INITIAL_LOSS_SCALE] [--min-loss-scale MIN_LOSS_SCALE] [--loss-scale-window LOSS_SCALE_WINDOW] [--hysteresis HYSTERESIS] [--fp32-residual-connection]
                       [--apply-query-key-layer-scaling] [--attention-softmax-in-fp32] [--accumulate-allreduce-grads-in-fp32] [--fp16-lm-cross-entropy] [--tensor-model-parallel-size TENSOR_MODEL_PARALLEL_SIZE]
                       [--encoder-tensor-model-parallel-size ENCODER_TENSOR_MODEL_PARALLEL_SIZE] [--pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE]
                       [--encoder-pipeline-model-parallel-size ENCODER_PIPELINE_MODEL_PARALLEL_SIZE] [--pipeline-model-parallel-split-rank PIPELINE_MODEL_PARALLEL_SPLIT_RANK]
                       [--decoder-first-pipeline-num-layers DECODER_FIRST_PIPELINE_NUM_LAYERS] [--decoder-last-pipeline-num-layers DECODER_LAST_PIPELINE_NUM_LAYERS] [--model-parallel-size MODEL_PARALLEL_SIZE]
                       [--num-layers-per-virtual-pipeline-stage NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE] [--num-virtual-stages-per-pipeline-rank NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK]
                       [--microbatch-group-size-per-virtual-pipeline-stage MICROBATCH_GROUP_SIZE_PER_VP_STAGE] [--no-overlap-p2p-communication] [--overlap-p2p-communication-warmup-flush] [--distributed-backend {nccl,gloo}]
                       [--distributed-timeout-minutes DISTRIBUTED_TIMEOUT_MINUTES] [--overlap-grad-reduce] [--defer-embedding-wgrad-compute] [--wgrad-deferral-limit WGRAD_DEFERRAL_LIMIT] [--no-align-grad-reduce]
                       [--ddp-num-buckets DDP_NUM_BUCKETS] [--ddp-bucket-size DDP_BUCKET_SIZE] [--ddp-pad-buckets-for-high-nccl-busbw] [--ddp-average-in-collective] [--overlap-param-gather]
                       [--overlap-param-gather-with-optimizer-step] [--no-align-param-gather] [--no-scatter-gather-tensors-in-pipeline] [--use-ring-exchange-p2p] [--local-rank LOCAL_RANK] [--lazy-mpu-init LAZY_MPU_INIT]
                       [--account-for-embedding-in-pipeline-split] [--account-for-loss-in-pipeline-split] [--use-distributed-optimizer] [--num-distributed-optimizer-instances NUM_DISTRIBUTED_OPTIMIZER_INSTANCES] [--use-torch-fsdp2]
                       [--context-parallel-size CONTEXT_PARALLEL_SIZE] [--cp-comm-type CP_COMM_TYPE [CP_COMM_TYPE ...]]
                       [--hierarchical-context-parallel-sizes HIERARCHICAL_CONTEXT_PARALLEL_SIZES [HIERARCHICAL_CONTEXT_PARALLEL_SIZES ...]] [--nccl-communicator-config-path NCCL_COMMUNICATOR_CONFIG_PATH] [--use-tp-pp-dp-mapping]
                       [--replication] [--replication-jump REPLICATION_JUMP] [--replication-factor REPLICATION_FACTOR] [--eval-iters EVAL_ITERS] [--eval-interval EVAL_INTERVAL] [--test-mode] [--skip-train]
                       [--data-path [DATA_PATH ...]] [--split SPLIT] [--train-data-path [TRAIN_DATA_PATH ...]] [--valid-data-path [VALID_DATA_PATH ...]] [--test-data-path [TEST_DATA_PATH ...]] [--data-args-path DATA_ARGS_PATH]
                       [--per-split-data-args-path PER_SPLIT_DATA_ARGS_PATH] [--data-cache-path DATA_CACHE_PATH] [--no-mmap-bin-files] [--mock-data] [--seq-length SEQ_LENGTH] [--encoder-seq-length ENCODER_SEQ_LENGTH]
                       [--decoder-seq-length DECODER_SEQ_LENGTH] [--retriever-seq-length RETRIEVER_SEQ_LENGTH] [--sample-rate SAMPLE_RATE] [--mask-prob MASK_PROB] [--short-seq-prob SHORT_SEQ_PROB] [--num-workers NUM_WORKERS]
                       [--reset-position-ids] [--reset-attention-mask] [--eod-mask-loss] [--no-create-attention-mask-in-dataloader] [--num-dataset-builder-threads NUM_DATASET_BUILDER_THREADS] [--s3-cache-path S3_CACHE_PATH]
                       [--vocab-size VOCAB_SIZE] [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--vocab-extra-ids VOCAB_EXTRA_IDS]
                       [--tokenizer-type {BertWordPieceLowerCase,BertWordPieceCase,GPT2BPETokenizer,SentencePieceTokenizer,GPTSentencePieceTokenizer,HuggingFaceTokenizer,Llama2Tokenizer,TikTokenizer,MultimodalTokenizer,NullTokenizer}]
                       [--tokenizer-model TOKENIZER_MODEL] [--tiktoken-pattern TIKTOKEN_PATTERN] [--tiktoken-num-special-tokens TIKTOKEN_NUM_SPECIAL_TOKENS]
                       [--tiktoken-special-tokens TIKTOKEN_SPECIAL_TOKENS [TIKTOKEN_SPECIAL_TOKENS ...]] [--adlr-autoresume] [--adlr-autoresume-interval ADLR_AUTORESUME_INTERVAL] [--ict-head-size ICT_HEAD_SIZE]
                       [--biencoder-projection-dim BIENCODER_PROJECTION_DIM] [--biencoder-shared-query-context-model] [--ict-load ICT_LOAD] [--bert-load BERT_LOAD] [--titles-data-path TITLES_DATA_PATH]
                       [--query-in-block-prob QUERY_IN_BLOCK_PROB] [--use-one-sent-docs] [--evidence-data-path EVIDENCE_DATA_PATH]
                       [--retriever-report-topk-accuracies RETRIEVER_REPORT_TOPK_ACCURACIES [RETRIEVER_REPORT_TOPK_ACCURACIES ...]] [--retriever-score-scaling] [--block-data-path BLOCK_DATA_PATH] [--embedding-path EMBEDDING_PATH]
                       [--indexer-batch-size INDEXER_BATCH_SIZE] [--indexer-log-interval INDEXER_LOG_INTERVAL] [--num-classes NUM_CLASSES] [--img-h IMG_H] [--img-w IMG_W] [--num-channels NUM_CHANNELS] [--patch-dim PATCH_DIM]
                       [--classes-fraction CLASSES_FRACTION] [--data-per-class-fraction DATA_PER_CLASS_FRACTION] [--no-data-sharding] [--head-lr-mult HEAD_LR_MULT] [--vision-pretraining]
                       [--vision-pretraining-type {classify,inpaint,dino}] [--vision-backbone-type {vit,mit,swin}] [--swin-backbone-type {tiny,base,h3}] [--mask-type {random,row}] [--mask-factor MASK_FACTOR]
                       [--iter-per-epoch ITER_PER_EPOCH] [--dino-local-img-size DINO_LOCAL_IMG_SIZE] [--dino-local-crops-number DINO_LOCAL_CROPS_NUMBER] [--dino-head-hidden-size DINO_HEAD_HIDDEN_SIZE]
                       [--dino-bottleneck-size DINO_BOTTLENECK_SIZE] [--dino-freeze-last-layer DINO_FREEZE_LAST_LAYER] [--dino-norm-last-layer] [--dino-warmup-teacher-temp DINO_WARMUP_TEACHER_TEMP]
                       [--dino-teacher-temp DINO_TEACHER_TEMP] [--dino-warmup-teacher-temp-epochs DINO_WARMUP_TEACHER_TEMP_EPOCHS] [--qk-layernorm] [--expert-model-parallel-size EXPERT_MODEL_PARALLEL_SIZE]
                       [--expert-tensor-parallel-size EXPERT_TENSOR_PARALLEL_SIZE] [--num-experts NUM_EXPERTS] [--moe-layer-freq MOE_LAYER_FREQ] [--moe-ffn-hidden-size MOE_FFN_HIDDEN_SIZE]
                       [--moe-shared-expert-intermediate-size MOE_SHARED_EXPERT_INTERMEDIATE_SIZE] [--moe-shared-expert-overlap] [--moe-grouped-gemm] [--moe-router-load-balancing-type {aux_loss,seq_aux_loss,sinkhorn,none}]
                       [--moe-router-score-function {softmax,sigmoid}] [--moe-router-topk MOE_ROUTER_TOPK] [--moe-router-pre-softmax] [--moe-router-num-groups MOE_ROUTER_NUM_GROUPS] [--moe-router-group-topk MOE_ROUTER_GROUP_TOPK]
                       [--moe-router-topk-scaling-factor MOE_ROUTER_TOPK_SCALING_FACTOR] [--moe-router-enable-expert-bias] [--moe-router-bias-update-rate MOE_ROUTER_BIAS_UPDATE_RATE] [--moe-use-legacy-grouped-gemm]
                       [--moe-aux-loss-coeff MOE_AUX_LOSS_COEFF] [--moe-z-loss-coeff MOE_Z_LOSS_COEFF] [--moe-input-jitter-eps MOE_INPUT_JITTER_EPS] [--moe-token-dispatcher-type {allgather,alltoall,flex,alltoall_seq}]
                       [--moe-enable-deepep] [--moe-per-layer-logging] [--moe-expert-capacity-factor MOE_EXPERT_CAPACITY_FACTOR] [--moe-pad-expert-input-to-capacity] [--moe-token-drop-policy {probs,position}] [--moe-layer-recompute]
                       [--moe-extended-tp] [--moe-use-upcycling] [--moe-permute-fusion] [--q-lora-rank Q_LORA_RANK] [--kv-lora-rank KV_LORA_RANK] [--qk-head-dim QK_HEAD_DIM] [--qk-pos-emb-head-dim QK_POS_EMB_HEAD_DIM]
                       [--v-head-dim V_HEAD_DIM] [--rotary-scaling-factor ROTARY_SCALING_FACTOR] [--mscale MSCALE] [--mscale-all-dim MSCALE_ALL_DIM] [--log-params-norm] [--log-num-zeros-in-grad] [--log-throughput] [--log-progress]
                       [--timing-log-level {0,1,2}] [--no-barrier-with-level-1-timing] [--timing-log-option {max,minmax,all}] [--tensorboard-log-interval TENSORBOARD_LOG_INTERVAL] [--tensorboard-queue-size TENSORBOARD_QUEUE_SIZE]
                       [--log-timers-to-tensorboard] [--no-log-loss-scale-to-tensorboard] [--log-validation-ppl-to-tensorboard] [--log-memory-to-tensorboard] [--log-world-size-to-tensorboard] [--wandb-project WANDB_PROJECT]
                       [--wandb-exp-name WANDB_EXP_NAME] [--wandb-save-dir WANDB_SAVE_DIR] [--logging-level LOGGING_LEVEL] [--log-straggler] [--disable-straggler-on-startup] [--straggler-ctrlr-port STRAGGLER_CTRLR_PORT]
                       [--straggler-minmax-count STRAGGLER_MINMAX_COUNT] [--inference-batch-times-seqlen-threshold INFERENCE_BATCH_TIMES_SEQLEN_THRESHOLD] [--max-tokens-to-oom MAX_TOKENS_TO_OOM] [--output-bert-embeddings]
                       [--bert-embedder-type {megatron,huggingface}] [--flash-decode] [--enable-cuda-graph] [--cuda-graph-warmup-steps CUDA_GRAPH_WARMUP_STEPS] [--inference-max-requests INFERENCE_MAX_BATCH_SIZE]
                       [--inference-max-seq-length INFERENCE_MAX_SEQ_LENGTH] [--fp8-format {e4m3,hybrid}] [--fp8-margin FP8_MARGIN] [--fp8-interval FP8_INTERVAL] [--fp8-amax-history-len FP8_AMAX_HISTORY_LEN]
                       [--fp8-amax-compute-algo {most_recent,max}] [--no-fp8-wgrad] [--transformer-impl {local,transformer_engine}] [--fp8-param-gather] [--te-rng-tracker] [--inference-rng-tracker]
                       [--retro-project-dir RETRO_PROJECT_DIR] [--retro-add-retriever] [--retro-cyclic-train-iters RETRO_CYCLIC_TRAIN_ITERS] [--retro-encoder-layers RETRO_ENCODER_LAYERS]
                       [--retro-encoder-hidden-dropout RETRO_ENCODER_HIDDEN_DROPOUT] [--retro-encoder-attention-dropout RETRO_ENCODER_ATTENTION_DROPOUT] [--retro-num-neighbors RETRO_NUM_NEIGHBORS]
                       [--retro-num-retrieved-chunks RETRO_NUM_RETRIEVED_CHUNKS] [--retro-attention-gate RETRO_ATTENTION_GATE] [--retro-no-verify-neighbor-count] [--spec [SPEC ...]] [--hybrid-attention-ratio HYBRID_ATTENTION_RATIO]
                       [--hybrid-mlp-ratio HYBRID_MLP_RATIO] [--hybrid-override-pattern HYBRID_OVERRIDE_PATTERN] [--yaml-cfg YAML_CFG] [--use-precision-aware-optimizer] [--main-grads-dtype {fp32,bf16}]
                       [--main-params-dtype {fp32,fp16}] [--exp-avg-dtype {fp32,fp16,fp8}] [--exp-avg-sq-dtype {fp32,fp16,fp8}] [--no-one-logger] [--one-logger-project ONE_LOGGER_PROJECT] [--one-logger-run-name ONE_LOGGER_RUN_NAME]
                       [--one-logger-async] [--app-tag-run-name APP_TAG_RUN_NAME] [--app-tag-run-version APP_TAG_RUN_VERSION] [--enable-ft-package] [--calc-ft-timeouts] [--config-logger-dir CONFIG_LOGGER_DIR]
                       [--error-injection-rate ERROR_INJECTION_RATE] [--error-injection-type {correct_result,transient_error,persistent_error}] [--rerun-mode {disabled,validate_results,report_stats}]

```

```
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py --micro-batch-size 1 --vocab-file vocab.json --merge-file merges.txt --save /home/user/Llama-3.1-8B-Instruct_Ajith --load /home/user/Llama-3.1-8B-Instruct --data-path wikitext_text_document --seq-length 1024 --max-position-embeddings 131072 --tokenizer-type GPT2BPETokenizer --tokenizer-model /home/user/Llama-3.1-8B-Instruct --exit-on-missing-checkpoint --use-checkpoint-args --no-load-optim --no-load-rng --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope --no-masked-softmax-fusion --attention-softmax-in-fp32 --disable-bias-linear --transformer-impl transformer_engine --group-query-attention 8 --attention-dropout 0.0 --hidden-dropout 0.0 --rotary-base 500000 --rotary-percent 1.0 --use-rope-scaling --ffn-hidden-size 14336 --num-attention-heads 32 --swiglu --bf16
```
