# global_eclektic_validation

```
uv venv eclektic
source eclektic/bin/activate
pip install -U pip
uv pip install torch transformers accelerate pandas huggingface_hub vllm
``` 

```
git clone https://github.com/catherinearnett/global_eclektic_validation.git
cd global_eclektic_validation
```
```
export HF_TOKEN_READ
export HF_TOKEN_WRITE
export HF_TOKEN_UPLOAD
export HF_TOKEN_MRL_READ
```

```
sample_and_filter.py
```
