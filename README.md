# global_eclektic_validation

```
uv venv eclektic
source eclektic/bin/activate
uv pip install torch transformers accelerate pandas huggingface_hub
``` 

```
git clone https://github.com/catherinearnett/global_eclektic_validation.git
cd global_eclektic_validation
```

export hf token
```
python3 model_generations.py
python3 sample_dataset.py
python3 upload.py
```
