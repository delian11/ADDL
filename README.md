# Deep Disturbance-Disentangled Learning for Facial Expression Recognition

This is the code repository for our paper "[Deep Disturbance-Disentangled Learning for Facial Expression Recognition](https://dl.acm.org/doi/abs/10.1145/3394171.3413907)" in ACM MM 2020.

![authors](/imgs/architecture.png)

## Train
- Pytorch
  
  Torch 1.1.0 or higher and Python 3.5 or higher are required.

- Prepare

    The tained DFEM models are reuqired. You can train them by yourself or use our trained models ([link](https://drive.google.com/drive/folders/1JUiB07GUvvRcxDowZePJH6cevk_9WfQt)) dirctly.


- Train DDM
```
python main.py --pretrain --dataset=RAF --bs=16
```
The pre-trained model on AffectNet dataset is available in [link](https://drive.google.com/drive/folders/1JUiB07GUvvRcxDowZePJH6cevk_9WfQt).


If you find our code or paper useful, please cite as

```
@inproceedings{DDL,
  title={Deep disturbance-disentangled learning for facial expression recognition},
  author={Ruan, Delian and Yan, Yan and Chen, Si and Xue, Jing-Hao and Wang, Hanzi},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2833--2841},
  year={2020}
}
```
