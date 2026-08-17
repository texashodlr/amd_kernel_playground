#!/bin/bash
#
docker run -it  \
    -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
    -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
    -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
    -e HSA_ENABLE_DXG_DETECTION=1 \
    --device=/dev/dxg \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 16G \
    rocm/pytorch:latest
