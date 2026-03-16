# Data Pipeline Configuration

## Data Directory Standards

### prepare.py Conventions
- Use numpy memmap for large datasets to avoid memory issues
- 90/10 train/val split is standard
- Save meta.pkl with vocab_size, encoder, decoder info
- Use uint16 for token IDs (sufficient for most vocabularies)

### Dataset Format
- Binary format: train.bin, val.bin
- Metadata: meta.pkl with pickle serialization
- Document dataset statistics in comments (vocab size, token count)

### Memory Management
- Recreate memmap per batch to avoid memory leaks
- Use dtype=np.uint16 for efficiency
- Pin memory for GPU transfer when possible
