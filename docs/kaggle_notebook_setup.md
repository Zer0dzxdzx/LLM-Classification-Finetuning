# Kaggle Notebook Setup

1. Open the competition page and create a new notebook.
2. Add the competition dataset as input.
3. Upload or copy this repository into `/kaggle/working`.
4. Confirm the data path exists:

```python
from pathlib import Path
Path("/kaggle/input/llm-classification-finetuning/train.csv").exists()
```

5. Install optional dependencies only if the environment does not already include them:

```bash
pip install -r requirements.txt
```

For Transformer finetuning:

```bash
pip install -r requirements-finetune.txt
```

If internet is disabled, add the pretrained model as a Kaggle dataset and set `model_name_or_path` in `configs/finetune.yaml` to that input path.
