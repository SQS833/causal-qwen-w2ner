# Causal Qwen W2NER

This project fine-tunes `Qwen/Qwen2.5-0.5B` with LoRA for the ocean NER corpus
used by `ocean.py`.  It preserves the corpus's W2NER grid encoding:

- `0`: no relation;
- `1`: NNW, an adjacent in-entity link (`i -> i+1`);
- `2+`: THW entity-tail to entity-head relation, with the entity type.

Unlike the original BERT W2NER implementation, the backbone is decoder-only and
keeps Qwen's causal attention mask.  Each grid relation is scored from the state
at its later token and the representation of its earlier token.  No bidirectional
encoder or unrestricted 2D convolution is used.

## Setup

The virtual environment is located at `.venv`.

```powershell
.\.venv\bin\Activate.ps1
.\.venv\bin\python.exe -m pip install -r requirements.txt
```

## Train

可以直接在 PyCharm 中打开并运行 `python/train.py`，不需要填写命令行参数。
脚本会自动完成“训练 → 每轮验证 → 保存最佳模型 → 测试集评估”一条龙流程。
默认使用项目下的 `data/ocean/`，结果保存到 `outputs/qwen-ocean/`。
PyCharm 的 Working directory 可以保持默认值。

也可以在终端运行：

```powershell
.\.venv\bin\python.exe python\train.py --data-dir data/ocean --output-dir outputs/qwen-ocean
```

The Python source files are kept under `python/`:

- `python/train.py`: LoRA fine-tuning entry point;
- `python/predict.py`: inference entry point;
- `python/src/`: data encoding, model, loss, and decoding modules.

## PyCharm interpreter

在 `Settings -> Project -> Python Interpreter` 中选择项目解释器：

```text
D:\HuaweiMoveData\Users\33140\Desktop\毕设\causal_qwen_w2ner\.venv\bin\python.exe
```

运行配置选择 `python/train.py` 即可一键训练。训练完成后，选择
`python/predict.py` 运行即可使用默认示例句子；也可以在 PyCharm 的
`Parameters` 中填写 `--checkpoint` 和 `--text`。

测试集文件名应为 `data/ocean/example.test`。如果目录中没有该文件，
训练脚本会自动使用 `example.dev` 作为测试集，并在终端打印提示。

## Server deployment

上传到 GitHub 时不要提交 `.venv/`、模型缓存和 `outputs/`。服务器端执行：

```bash
git clone <你的仓库地址>
cd causal_qwen_w2ner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python python/train.py --data-dir data/weibo --output-dir outputs/qwen-weibo
```

如果服务器没有外网，先将 Qwen 模型下载到服务器，并使用本地路径：

```bash
python python/train.py --model-name /path/to/Qwen2.5-0.5B
```

The first run downloads `Qwen/Qwen2.5-0.5B` from Hugging Face.  Set
`--model-name` to a local Qwen checkpoint if internet access is unavailable.

## Design

For a grid cell `(i, j)`, the model uses the state at `max(i, j)`, so a predicted
relation never depends on tokens after the later endpoint.  The `--lookahead`
option adds a fixed, causal confirmation delay for THW/end predictions: the
decision for end `e` may use the state at `e + delta`, never unrestricted future
context.  Use `--lookahead 0` for strict online inference.
