sudo docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --mount type=bind,source=/home/ubuntu,target=/gpu \
    --shm-size 8G \
    rocm/pytorch:latest