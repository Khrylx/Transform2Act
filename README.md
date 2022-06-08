# Transform2Act
This repo contains the official implementation of our paper:
  
Transform2Act: Learning a Transform-and-Control Policy for Efficient Agent Design  
Ye Yuan, Yuda Song, Zhengyi Luo, Wen Sun, Kris Kitani  
**ICLR 2022 (Oral)**  
[website](https://sites.google.com/view/transform2act) | [paper](https://openreview.net/forum?id=UcDUxjPYWSr)

<img src="assets/media/teaser.png" width="800">

<img src="assets/media/design_process_2dloc.gif" width="400"> <img src="assets/media/design_process_3dloc.gif" width="400">

# Installation 

### Environment
* **Tested OS:** MacOS, Linux
* Python >= 3.7
* PyTorch == 1.8.0
### Dependencies:
1. Install [PyTorch 1.8.0](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Install torch-geometric with correct CUDA and PyTorch versions (change the `CUDA` and `TORCH` variables below): 
    ```
    CUDA=cu102
    TORCH=1.8.0
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-geometric==1.6.1
    ```
4. install mujoco-py following the instruction [here](https://github.com/openai/mujoco-py#install-mujoco).
5. Set the following environment variable to avoid problems with multiprocessed sampling:    
    ```
    export OMP_NUM_THREADS=1
    ```

### Pretrained Models
* You can download pretrained models from [Google Drive](https://drive.google.com/file/d/1-pJrGPCcbaiCpENss5jYzRF_ZFJncFJB/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1szh3F97T9JNhoV2rG_-Gew?pwd=2x3q) (password: 2x3q).
* Once the `transform2act_models.zip` file is downloaded, unzip it under the `results` folder of this repo:
  ```
  mkdir results
  unzip transform2act_models.zip -d results
  ```
  Note that the pretrained models directly correspond to the config files in [design_opt/cfg](design_opt/cfg).

# Training
You can train your own models using the provided config in [design_opt/cfg](design_opt/cfg):
```
python design_opt/train.py --cfg hopper --gpu 0
```
You can replace `hopper` with {`ant`, `gap`, `swimmer`} to train other environments. Here is the correspondence between the configs and the environments in the paper: `hopper - 2D Locomotion`, `ant - 3D Locomotion`, `swimmer - Swimmer`, and `gap - Gap Crosser`.

# Visualization
If you have a display, run the following command to visualize the pretrained model for the `hopper`:
```
python design_opt/eval.py --cfg hopper
```
Again, you can replace `hopper` with {`ant`, `gap`, `swimmer`} to visualize other environments.

You can also save the visualization into a video by using `--save_video`:
```
python design_opt/eval.py --cfg hopper --save_video
```
This will produce a video `out/videos/hopper.mp4`.

# Citation
If you find our work useful in your research, please cite our paper [Transform2Act](https://sites.google.com/view/transform2act/):
```
@inproceedings{yuan2022transform2act,
  title={Transform2Act: Learning a Transform-and-Control Policy for Efficient Agent Design},
  author={Yuan, Ye and Song, Yuda and Luo, Zhengyi and Sun, Wen and Kitani, Kris},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

# License
Please see the [license](LICENSE) for further details.
