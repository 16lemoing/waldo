# Data preprocessing

We use [mmflow](https://github.com/open-mmlab/mmflow) to generate flow maps and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to generate semantic segmentation maps.

We recommand using a new environment to install these libraries. All the corresponding steps are described [here](https://github.com/open-mmlab/mmflow/blob/master/docs/en/install.md#prepare-environment) and [here](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation).

For extracting the flow we use [this RAFT config](https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.py), and for extracting segmentation maps we use [this DeepLabV3 config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py). Other choices of model and config should also work.