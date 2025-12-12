# DouZero - 斗地主 AI 强化学习框架

## 项目概述

DouZero 是一个为斗地主（DouDizhu）设计的强化学习框架，由快手 AI 平台部开发。该项目在 ICML 2021 上发表，通过自我博弈深度强化学习来掌握斗地主游戏。

**核心特点：**
- 针对斗地主这一具有挑战性的领域：包含合作、竞争、非完全信息、庞大状态空间
- 使用深度蒙特卡洛（DMC）算法，结合动作编码和并行执行器
- 支持 GPU 和 CPU 训练（Windows 用户可使用 CPU 训练）

**项目类型：** Python 强化学习项目

## 技术栈

- **编程语言:** Python 3.6+
- **深度学习框架:** PyTorch (>=1.6.0)
- **依赖库:** 
  - `torch` - 深度学习框架
  - `rlcard` - 卡牌游戏环境
  - `GitPython` - Git 操作
  - `gitdb2` - Git 数据库

## 项目结构

```
douzero/
├── douzero/                    # 核心代码
│   ├── dmc/                   # 深度蒙特卡洛算法实现
│   │   ├── arguments.py      # 参数解析
│   │   ├── dmc.py           # DMC 主逻辑
│   │   ├── env_utils.py     # 环境工具
│   │   ├── file_writer.py   # 文件写入
│   │   ├── models.py        # 神经网络模型
│   │   └── utils.py         # 工具函数
│   ├── env/                  # 游戏环境
│   │   ├── env.py           # 环境主类
│   │   ├── game.py          # 游戏逻辑
│   │   ├── move_detector.py # 动作检测
│   │   ├── move_generator.py # 动作生成
│   │   ├── move_selector.py # 动作选择
│   │   └── utils.py         # 环境工具
│   └── evaluation/          # 评估模块
│       ├── deep_agent.py    # 深度智能体
│       ├── random_agent.py  # 随机智能体
│       ├── rlcard_agent.py  # RLCard 智能体
│       └── simulation.py    # 模拟评估
├── baselines/               # 预训练模型目录
├── imgs/                   # 图片资源
├── evaluate.py             # 评估脚本
├── generate_eval_data.py   # 生成评估数据
├── get_most_recent.sh      # 获取最新模型脚本
├── requirements.txt        # Python 依赖
├── setup.py               # 包配置
└── train.py               # 训练脚本
```

## 安装与设置

### 环境要求
- Python 3.6+
- CUDA（如需 GPU 训练）
- Git

### 安装步骤

1. **克隆仓库：**
   ```bash
   git clone https://github.com/kwai/DouZero.git
   cd DouZero
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装稳定版本（推荐）：**
   ```bash
   pip install douzero
   ```

   或安装开发版本：
   ```bash
   pip install -e .
   ```

## 训练模型

### 基本训练（单 GPU）
```bash
python train.py
```

### 多 GPU 训练
```bash
python train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```

### CPU 训练（Windows 用户）
```bash
python train.py --actor_device_cpu --training_device cpu
```

### 训练参数说明
- `--gpu_devices`: 可用的 GPU 设备
- `--num_actor_devices`: 用于模拟的 GPU 数量
- `--num_actors`: 每个设备的执行器数量
- `--training_device`: 用于训练的 GPU 索引（或 "cpu"）
- `--actor_device_cpu`: 使用 CPU 作为执行器
- `--objective {adp,wp}`: 使用 ADP（平均分差）或 WP（胜率）作为奖励
- `--total_frames`: 训练的总帧数
- `--batch_size`: 学习器批大小
- `--learning_rate`: 学习率

## 评估模型

### 1. 生成评估数据
```bash
python generate_eval_data.py
```

可选参数：
- `--output`: 保存 pickle 数据的路径
- `--num_games`: 生成的随机游戏数量（默认 10000）

### 2. 运行评估
```bash
python evaluate.py
```

### 评估配置示例

**地主位置评估：**
```bash
python evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```

**农民位置评估：**
```bash
python evaluate.py --landlord rlcard --landlord_up baselines/douzero_ADP/landlord_up.ckpt --landlord_down baselines/douzero_ADP/landlord_down.ckpt
```

### 评估参数
- `--landlord`: 地主智能体（random、rlcard 或模型路径）
- `--landlord_up`: 上家农民智能体
- `--landlord_down`: 下家农民智能体
- `--eval_data`: 评估数据 pickle 文件
- `--num_workers`: 子进程数量
- `--gpu_device`: GPU 设备（默认使用 CPU）

## 预训练模型

预训练模型可从以下位置获取：
- Google Drive: https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB
- 百度网盘: https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ (提取码: 4624)

将模型放置在 `baselines/` 目录下：
- `baselines/douzero_ADP/`: ADP 目标预训练模型
- `baselines/douzero_WP/`: WP 目标预训练模型
- `baselines/sl/`: 基于人类数据的预训练模型

## 开发约定

### 代码结构
- 核心算法在 `douzero/dmc/` 目录中
- 游戏环境在 `douzero/env/` 目录中
- 评估模块在 `douzero/evaluation/` 目录中

### 模型保存
- 模型默认保存在 `douzero_checkpoints/douzero/` 目录
- 每半小时自动保存检查点
- 使用 `get_most_recent.sh` 脚本获取最新模型

### 平台注意事项
- **Windows:** 仅支持 CPU 训练（GPU 多进程不支持）
- **Linux/macOS:** 支持 GPU 和 CPU 训练

## 常用命令参考

### 获取最新模型
```bash
sh get_most_recent.sh douzero_checkpoints/douzero/
```

### 检查模型目录
```bash
ls -la baselines/
```

### 运行测试游戏
```bash
python generate_eval_data.py --num_games 100
python evaluate.py --eval_data eval_data.pkl --num_workers 2
```

## 故障排除

### Windows GPU 问题
Windows 系统在 GPU 训练时可能出现 "operation not supported" 错误，这是因为 Windows 不支持 CUDA 张量的多进程操作。解决方案：
- 使用 CPU 训练：`python train.py --actor_device_cpu --training_device cpu`
- 或使用 Linux/macOS 系统

### 依赖安装问题
如果 pip 安装缓慢，可使用清华镜像：
```bash
pip install douzero -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 内存不足
减少批大小或执行器数量：
```bash
python train.py --batch_size 32 --num_actors 5
```

## 扩展资源

- **在线演示:** https://www.douzero.org/
- **论文:** https://arxiv.org/abs/2106.06135
- **GitHub:** https://github.com/kwai/DouZero
- **中文文档:** README.zh-CN.md
- **社区:** Slack 频道和 QQ 群（详见 README）

---

*最后更新: 2025-12-12*  
*基于项目 README 和代码结构分析生成*