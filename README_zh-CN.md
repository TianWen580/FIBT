<div align="center">
    <h2>
        FIBTNet: Building Change Detection for Remote Sensing images Using Feature Interactive Bi-temporal Network
    </h2>
</div>
<br>

[![GitHub stars](https://badgen.net/github/stars/TianWen580/FIBT)](https://github.com/TianWen580/FIBT)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

<br>
<br>

<div align="center">

<!-- English | [简体中文](README_zh-CN.md) -->
[English](README.md) | 简体中文

</div>

## 介绍

![fibt-img](docs/fibt.png)

本仓库包含论文《FIBTNet: Building Change Detection for Remote Sensing images Using Feature Interactive Bi-temporal Network》的代码，该论文基于MMSegmentation和Open-CD框架。

如果您觉得本项目对您有帮助，请给我们一个星⭐️。您的支持是我们最大的动力。

## 依赖

FIBTNet基于Open-CD工具箱。Open-CD是一个基于一系列开源通用视觉任务工具的开源变化检测工具箱。

该分支的当前版本已验证可在PyTorch 2.x和CUDA 12.1上运行，适用于Python 3.7及以上版本，并与各种CUDA版本兼容。

## 新闻
🌟 2024年6月6日 - FIBTNet公开发布！

## 待办事项

- [ ] 支持单图像演示
- [ ] 支持
- [ ] ...

## 使用

### 依赖项
- Linux或Windows
- Python 3.7以上，推荐3.10
- PyTorch 2.0或更高，推荐2.1
- CUDA 11.7或更高，推荐12.1
- MMCV 2.0或更高，推荐2.1

### 环境安装

我们建议使用Miniconda进行安装。以下命令将创建一个名为`ttp`的虚拟环境并安装PyTorch和MMCV。

> 注意：如果您有PyTorch的使用经验并已安装，可以跳过此部分。否则，您可以按照这些步骤进行安装。

```shell
conda create -n fibt python=3.10 -y
conda activate fibt
```

然后，在Linux/Windows上安装PyTorch：

```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

或

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 依赖项安装

安装`MMCV`

```shell
pip install -U openmim
mim install mmcv==2.1.0
```

安装其他依赖项。

```shell
pip install -U wandb einops importlib peft==0.8.2 scipy ftfy prettytable torchmetrics==1.3.1 transformers==4.38.1
```

### 安装FIBTNet

下载或克隆`fibt`仓库。

```shell
git clone git@github.com:TianWen580/fibt.git
cd fibt
```

## 数据集准备

### Levir-CD变化检测数据集

#### 数据集下载

- 图像和标签下载地址：[Levir-CD](https://chenhao.in/LEVIR/)。

#### 组织方式

您也可以选择其他来源下载数据，但需要按照以下格式组织数据集：

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/levir-cd
├── train
│   ├── A
│   ├── B
│   └── label
├── val
│   ├── A
│   ├── B
│   └── label
└── test
    ├── A
    ├── B
    └── label
```

注意：在项目文件夹中，我们提供了一个名为`data`的文件夹，其中包含上述数据集组织方式的示例。

### 其他数据集

如果您想使用其他数据集，可以参考[MMSegmentation文档](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html)准备数据集。
</details>

## 模型训练

### FIBTNet模型

#### 配置文件和主要参数解析

我们提供了论文中使用的FIBTNet模型的配置文件，位于`configs/fibt`文件夹中。配置文件与MMSegmentation的API接口和用法完全一致。下面我们提供了一些主要参数的解析。如果您想了解更多参数含义，可以参考[MMSegmentation文档](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html)。

#### 单卡训练

```shell
python tools/train.py configs/fibt/xxx.py  # xxx.py 是您想使用的配置文件
```

#### 多卡训练

```shell
sh ./tools/dist_train.sh configs/fibt/xxx.py ${GPU_NUM}  # xxx.py 是您想使用的配置文件，GPU_NUM 是使用的GPU数量
```
## 模型测试

#### 单卡测试：

```shell
python tools/test.py configs/fibt/xxx.py ${CHECKPOINT_FILE}  # xxx.py 是您想使用的配置文件，CHECKPOINT_FILE 是您想使用的检查点文件
```

#### 多卡测试：

```shell
sh ./tools/dist_test.sh configs/fibt/xxx.py ${CHECKPOINT_FILE} ${GPU_NUM}  # xxx.py 是您想使用的配置文件，CHECKPOINT_FILE 是您想使用的检查点文件，GPU_NUM 是使用的GPU数量
```

**注意**：如果您需要获取可视化结果，可以在配置文件中取消注释`default_hooks-visualization`。

## 图像预测

#### 单图像预测：

待办...

## 引用

如果您在研究中发现此项目有用，请引用：

<!-- ```bibtex
@ARTICLE{10438490,
  author={Li, Kaiyu and Cao, Xiangyong and Meng, Deyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A New Learning Paradigm for Foundation Model-based Remote Sensing Change Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Adaptation models;Task analysis;Data models;Computational modeling;Feature extraction;Transformers;Tuning;Change detection;foundation model;visual tuning;remote sensing image processing;deep learning},
  doi={10.1109/TGRS.2024.3365825}}

@ARTICLE{10129139,
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2023.3277496}}
``` -->

## 许可证

FIBTNet在Apache 2.0许可证下发布。