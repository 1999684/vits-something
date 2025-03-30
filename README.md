# VITS 语音合成系统

> 本项目属于外行人编辑，且为上传中文相关内容和前端代码。

基于 VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) 的端到端文本到语音合成系统。本项目实现了单说话人和多说话人的语音合成功能，并提供了 Web API 接口。

## 功能特点

- 端到端的文本到语音合成
- 支持单说话人和多说话人模式
- 基于变分推理和对抗学习的高质量语音生成
- RESTful API 接口，便于集成到其他应用
- 简洁的 Web 服务，支持在线语音合成

## 安装步骤

### 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 支持 (推荐用于训练)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载预训练模型
下载后将模型文件放置在 pretrained_models 目录下。

## 使用方法
### 启动 Web 服务
```
python app.py
```

服务将在 http://localhost:5000 启动。

### API 接口 单说话人语音合成
- 端点 : /synthesize
- 方法 : POST
- 参数 :
  - text : 要合成的文本
- 示例 :
  ```
  curl -X POST http://localhost:5000/synthesize -H "Content-Type: application/json" -d "{\"text\":\"这是一段测试文本\"}"
  ```
### 多说话人语音合成
- 端点 : /synthesize_with_speaker
- 方法 : POST
- 参数 :
  - text : 要合成的文本
  - speaker_id : 说话人 ID
- 示例 :
  ```bash
  curl -X POST http://localhost:5000/synthesize_with_speaker -H "Content-Type: application/json" -d "{\"text\":\"这是一段测试文本\",\"speaker_id\":0}"
  ```
### 获取说话人列表
- 端点 : /get_speakers
- 方法 : GET
- 示例 :
  ```bash
  curl http://localhost:5000/get_speakers
  ```
## 命令行推理
### 单说话人推理
```bash
python inference.py --text "要合成的文本" --model_path "pretrained_models/single_speaker_model.pth" --config_path "configs/ljs_base.json" --output_wav "output.wav"
```
### 多说话人推理
```bash
python inference.py --text "要合成的文本" --model_path "pretrained_models/multi_speaker_model.pth" --config_path "configs/vctk_base.json" --output_wav "output.wav" --speaker_id 0
```
## 模型训练
### 单说话人训练
```bash
python train.py -c configs/ljs_base.json -m ljs_model
```

### 多说话人训练
```bash
python train_ms.py -c configs/vctk_base.json -m vctk_model
```

## 项目结构
```plaintext
vits/
├── app.py                # Web 服务
├── inference.py          # 推理代码
├── models.py             # 模型定义
├── modules.py            # 模型组件
├── attentions.py         # 注意力机制
├── commons.py            # 通用工具函数
├── transforms.py         # 变换函数
├── train.py              # 单说话人训练脚本
├── train_ms.py           # 多说话人训练脚本
├── utils.py              # 工具函数
├── text/                 # 文本处理
│   ├── __init__.py       # 文本到序列转换
│   ├── symbols.py        # 文本符号集
│   └── cleaners.py       # 文本清洗
├── configs/              # 配置文件
│   ├── ljs_base.json     # LJSpeech 配置
│   └── vctk_base.json    # VCTK 配置
└── pretrained_models/    # 预训练模型目录
```

## 致谢
- 感谢 [jaywalnut310/vits](https://github.com/jaywalnut310/vits) 提供的原始实现