# Megatron_Training

# Megatron_Traininig

#### Pull Docker container 
docker run -it --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH --security-opt seccomp=unconfined \ 
    -v /mnt/Scratch_space/ajith:/home/user -w /home/user nvcr.io/nvidia/pytorch:25.02-py3 bash
