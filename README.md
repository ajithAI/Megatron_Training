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

python3 -m torch.distributed.launch \
        pretrain_gpt.py \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH_New \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH
```
```
python3 -m torch.distributed.launch pretrain_gpt.py \
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 1 \
--seq-length 8192 \
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
