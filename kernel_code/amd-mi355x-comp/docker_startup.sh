git clone https://github.com/ROCm/aiter.git
git clone https://github.com/AMD-AIM/reference-kernels.git
export PYTHONPATH=$PYTHONPATH:/aiter/
reference-kernels/problems/amd_202602/mixed-mla/
apt-get update
apt install vim
cp ../eval.py .
cp ../utils.py .
pip install psutil pybind11 