# Data

## Dataset: NC_Physics

The NC_Physics dataset contains 10,200 peer-reviewed scientific papers from Nature Communications, covering diverse physics-related domains (April 2010 - March 2025).

### Download

<!-- TODO: Add download link -->
Dataset will be released soon.

### Expected Files

After downloading, place the following files in this directory:

```
data/
├── NC_Physics_train.jsonl      # Training data
├── NC_Physics_test.jsonl       # Test data 
└── README.md                   # This file
```

### Data Format

Each line in the JSONL files is a JSON object with:
- `title`: Paper title
- `introduction`: Ground-truth introduction section
- `methods`: Methods section
- `results`: Results section
- `analyses`: Analysis section
- `references`: List of cited references
- `core_idea`: Core research idea (for graph evaluation)
- `entities`: Key entities (for graph evaluation)

### Preprocessing

To convert raw data to training format:

```bash
python data_preprocess/NC_Physics.py --local_dir ./data/ --name LECTOR
```

This produces:
- `data/LECTOR_train.parquet` - Training split
- `data/LECTOR_val.parquet` - Validation split
