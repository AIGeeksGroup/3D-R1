# <img src="./assets/3dr1_logo.png" alt="logo" width="25"/> 3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding

This is the official repository for the paper:
> **3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding**
>
> [Ting Huang](https://github.com/Believeht029)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2507.23478) | [Website](https://aigeeksgroup.github.io/3D-R1) | [Data](https://huggingface.co/datasets/AIGeeksGroup/Scene-30K) | [Models](https://huggingface.co/AIGeeksGroup/3D-R1) | [HF Paper](https://huggingface.co/papers/2507.23478)

> [!NOTE]
> 💪 This and following visualizations show the **zero-shot** results of 3D-R1 in various complex scenes, demonstrating its incredible generalizability and state-of-the-art performance.

https://github.com/user-attachments/assets/45fa0d43-d30e-4211-9194-defd60d8f9c4

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{huang20253d,
  title={3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding},
  author={Huang, Ting and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2507.23478},
  year={2025}
}
```
---

## 🏃 Intro 3D-R1
3D-R1 is an open-source **generalist** model that enhances the reasoning of 3D VLMs for unified scene understanding.

Large vision-language models (VLMs) have made significant strides in 2D visual understanding tasks, sparking interest in extending these capabilities to 3D scene understanding.
However, current 3D VLMs often struggle with robust reasoning and generalization due to limitations in high-quality spatial data and the static nature of viewpoint assumptions.
To address these challenges, we propose **3D-R1**, a foundation model that enhances the reasoning capabilities of 3D VLMs.
Specifically, we first construct a high-quality synthetic dataset with CoT, named Scene-30K, leveraging existing 3D-VL datasets and a data engine based on Gemini 2.5 Pro. It serves as cold-start initialization data for 3D-R1.
Moreover, we leverage RLHF policy such as GRPO in the reinforcement learning training process to enhance reasoning capabilities and introduce three reward functions: a perception reward, a semantic similarity reward and a format reward to maintain detection accuracy and answer semantic precision.
Furthermore, we introduce a dynamic view selection strategy that adaptively chooses the most informative perspectives for 3D scene understanding.
Extensive experiments demonstrate that 3D-R1 delivers an average improvement of 10\% across various 3D scene benchmarks, highlighting its effectiveness in enhancing reasoning and generalization in 3D scene understanding.

![image](./assets/structure.png)

## 📰 News
<b>2025/08/07:</b> 🎉 Our paper has been shared by <a href="https://mp.weixin.qq.com/s/Feh7S4AJOmxzDx0gRlYyAw"><b>Deep Blue AI</b></a>.

<b>2025/08/05:</b> 🎉 Our paper has been shared by <a href="https://x.com/_akhaliq/status/1952533689583181932"><b>AK</b></a>.

<b>2025/08/04:</b> 📌 Our paper has been promoted by <a href="https://mp.weixin.qq.com/s/TgFY_hZcA7tKX163kztHXg"><b>AIxiv</b></a>.

<b>2025/08/03:</b> 🔔 Our paper has been promoted by <a href="https://mp.weixin.qq.com/s/3iLqHzfP8IEv4m5ln6_PQQ"><b>Learn AI with us</b></a>.

<b>2025/08/01:</b> 📣 Our paper has been promoted by <a href="https://zhuanlan.zhihu.com/p/1934643503331256268"><b>52CV</b></a>.

## TODO List

> [!IMPORTANT] 
> **General Response to Visualization:**
> We acknowledge that some users are seeking detailed visualization code. Regarding the bounding box drift issue in the visualization, we are currently fixing it and will update the visualization results accordingly, along with releasing a detailed visualization tutorial.


- [x] Upload our paper to arXiv and build project pages.
- [x] Upload the code.
- [x] Release Scene-30K dataset. (see [Scene-30K](https://huggingface.co/datasets/AIGeeksGroup/Scene-30K))
- [x] Release RL part code.
- [ ] Release visualization script.
- [ ] Add a demo on huggingface.


## ![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white) YouTube Video

>[!NOTE]
> If you’d like to learn more about our paper, be sure to check out this [**youtube video**](https://youtu.be/P3jPg6MwLOM) by @AIResearchRoundup.

[![Watch the video](https://img.youtube.com/vi/P3jPg6MwLOM/maxresdefault.jpg)](https://youtu.be/P3jPg6MwLOM)


## ⚡ Quick Start
### Environment Setup

Our code is tested with CUDA 11.8 and Python 3.9.16. To run the codes, you should first install the following packages:
```
h5py
scipy
cython
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
'torch=2.0.1+cu118'
google-generativeai
peft>=0.7.0
transformers>=4.35.0
accelerate>=0.20.0
tqdm
orjson
clip @ git+https://github.com/openai/CLIP.git
git+https://github.com/LiheYoung/Depth-Anything.git
```
After that, build the `pointnet2` and accelerated `giou` from source:
```bash
# PointNet++
cd third_party/pointnet2
python setup.py install

cd utils
python cython_compile.py build_ext --inplace
```
### Data Preparation
#### Download and Prepare the ScanNet 3D Data
You can download the pre-processed data from [here](https://huggingface.co/datasets/AIGeeksGroup/ScanQA).
Process 3D data: Follow the instructions [here](https://github.com/daveredrum/Scan2Cap/blob/main/data/scannet/README.md) and download the ScanNetV2 dataset.

#### Prepare Language Annotations

To train the model, you are required to prepare language annotations from `ScanRefer`, `Nr3D`, `ScanQA`, and the ScanNet part of `3D-LLM`.

1. `ScanRefer`. Follow the commands [here](https://github.com/daveredrum/ScanRefer) to download the `ScanRefer` dataset.
2. `Nr3D`. Follow the commands [here](https://referit3d.github.io/#dataset) to download the `Nr3D` dataset.
3. `ScanQA`. Follow the commands [here](https://github.com/ATR-DBI/ScanQA/blob/main/docs/dataset.md) to download the `ScanQA` dataset.
4. `3D-LLM`. The data are located at [here](https://vis-www.cs.umass.edu/3dllm/).

#### Scene-30K synthetic
You can synthesize Scene-30K by:
```bash
bash script/synthesize_scene30K.sh
```
Or you can download from [huggingface](https://huggingface.co/datasets/AIGeeksGroup/Scene-30K)

#### Download Pre-trained LLM weights
If your server has no trouble auto-downloading weights from huggingface🤗, feel free to skip this step.

Download files from the `Qwen2.5-VL-7B-Instruct` checkpoint at [huggingface](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## 💻 Train your own models
### SFT Training
We provide training script in the `script` folder with different LLM backends. Feel free to modify the hyper parameters in those commands.
SFT on Scene-30K as a cold-start:
```bash
bash script/train.generalist.sh
```
### RL Training
```bash
bash script/train.rl.sh
```

## 👩🏻‍💻 Case Study

<table>
  <tr>
    <td  align="center" valign="top">
      <b>3D Scene Dense Captioning (3D-DC)</b>
      <video src="https://github.com/user-attachments/assets/34ebbb05-6fc2-4c9e-a957-955c54edaf70"  controls></video><br>
    </td>
    <td  align="center" valign="top">
      <b>3D Object Captioning</b>
      <video src="https://github.com/user-attachments/assets/87d70ed8-aed5-48a7-9e2c-fb35464f76d9"  controls></video><br>
    </td>
    </tr>
  <tr>
    <td  align="center" valign="top">
      <b>3D Visual Grounding (3D-VG)</b>
      <video src="https://github.com/user-attachments/assets/703fe056-7a82-4921-9768-208b6d1dd9a0"  controls></video><br>
    </td>
    <td  align="center" valign="top">
      <b>3D Question Answering (3D-QA)</b>
      <video src="https://github.com/user-attachments/assets/3448dc2b-a015-4169-a9ea-df21289b2f63"  controls></video><br>
    </td>
    </tr>
  <tr>
    <td  align="center" valign="top">
      <b>3D Dialogue</b>
      <video src="https://github.com/user-attachments/assets/bf17fbb1-a3ef-4013-8526-f3495bc9fd35"  controls></video><br>
    </td>
    <td  align="center" valign="top">
      <b>3D Reasoning</b>
      <video src="https://github.com/user-attachments/assets/5eb4b5c8-5175-4751-91d6-7e936a47d8a2"  controls></video><br>
    </td>
    </tr>
  <tr>
    <td  align="center" valign="top">
      <b>3D Planning</b>
      <video src="https://github.com/user-attachments/assets/c41d3267-03b2-43f6-9102-261172c680b1"  controls></video><br>
    </td>
  </tr>
</table>

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/3D-R1&type=Date)](https://www.star-history.com/#AIGeeksGroup/3D-R1&Date)


## 😘 Acknowledgement
We thank the authors of [Qwen](https://github.com/QwenLM/Qwen), [LSceneLLM](https://github.com/Hoyyyaard/LSceneLLM), [ARKit](https://github.com/apple/ARKitScenes), and [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) for their open-source code.
