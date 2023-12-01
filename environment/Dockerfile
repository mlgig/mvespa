# hash:sha256:48c45cd94b6b946892aba329dead901eb1541e085b52f10459f32f408edf3a2c
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
        matplotlib==3.8.0 \
        numpy==1.26.0 \
        pandas==2.1.1 \
        scikit-learn==1.3.0 \
        scipy==1.11.3 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    mat4py==0.6.0 \
    nbformat \
    pingouin==0.5.3 \
    statsmodels==0.14.0 \
    tqdm==4.66.1
