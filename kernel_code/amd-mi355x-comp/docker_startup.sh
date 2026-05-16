#!/bin/bash
echo "Cloning AITER Repo"
git clone https://github.com/ROCm/aiter.git
sleep 1
echo "Cloning reference kernels repo"
git clone https://github.com/AMD-AIM/reference-kernels.git
sleep 1
echo "Exporting path"
export PYTHONPATH=$PYTHONPATH:/aiter/
sleep 1
echo "Changing directories"
cd reference-kernels/problems/amd_202602/mixed-mla/
echo "Resolving dependencies"
apt-get update
apt install vim
cp ../eval.py .
cp ../utils.py .
pip install psutil pybind11 
echo "Ready for reference kernel build"
sleep 1

exit