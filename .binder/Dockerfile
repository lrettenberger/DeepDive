FROM lrettenberger/deepdive:latest

USER root


 RUN conda install --yes \
    'python=3.9.18' \

RUN pip install\
    'torch==2.1.2' \
    'pytorch-lightning==2.1.1' \
    'torchvision==0.16.2' \
    'torchmetrics==1.2.0' \
    'matplotlib' \
    'seaborn' \
    'requests' \
    'torchsummary'

USER $NB_UID