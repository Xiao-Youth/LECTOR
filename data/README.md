# Data

## Dataset: NC_Physics

The NC_Physics dataset is released on Hugging Face at [Xiao-Youth/NC_Physics](https://huggingface.co/datasets/Xiao-Youth/NC_Physics). It contains peer-reviewed Nature Communications papers covering diverse physics-related domains.

The released files are:
- `NC_Physics_trainval.jsonl`: 11,341 examples. The default LECTOR preprocessing uses 10,000 examples for training and 100 examples for validation from this file.
- `NC_Physics_test.jsonl`: 100 test examples.

### Download

From the repository root:

```bash
hf download Xiao-Youth/NC_Physics \
    --repo-type dataset \
    --local-dir data \
    --include "NC_Physics_*.jsonl"
```

### Expected Files

After downloading, place the following files in this directory:

```
data/
├── NC_Physics_trainval.jsonl   # Raw train/validation pool
├── NC_Physics_test.jsonl       # Test data
└── README.md                   # This file
```

### Data Format

Each line in the JSONL files is a JSON object with:
- `unique_id`: Paper identifier
- `subfield`: Scientific subfield
- `title`: Paper title
- `abstract`: Paper abstract
- `sections`: Paper sections. The first section is the ground-truth Introduction; default preprocessing excludes it from the input and uses it as the target text.
- `references`: List of cited references
- `core_idea`: Core research idea (for graph evaluation)
- `entities`: Key entities (for graph evaluation)

### Preprocessing

To convert raw data to training format with the default 10,000/100 train/validation split:

```bash
python data_preprocess/NC_Physics.py \
    --local_dir ./data/ \
    --name LECTOR \
    --train_size 10000 \
    --val_size 100
```

This produces:
- `data/LECTOR_train.parquet` - Training split
- `data/LECTOR_val.parquet` - Validation split

### Model Checkpoint

The released LECTOR-4B checkpoint is available at [Xiao-Youth/LECTOR-4B](https://huggingface.co/Xiao-Youth/LECTOR-4B):

From the repository root:

```bash
hf download Xiao-Youth/LECTOR-4B \
    --local-dir ckpts/LECTOR-4B
```
