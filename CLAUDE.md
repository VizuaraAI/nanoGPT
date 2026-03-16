# nanoGPT Project Configuration

## ML-Specific Coding Standards

### Tensor Shape Documentation
- ALWAYS document tensor shapes in comments: `# (B, T, C)` format
- Use `B` for batch, `T` for sequence/time, `C` for channels/embedding dim
- Include shape comments for all forward() method inputs/outputs

### Device-Agnostic Code
- Never hardcode `.cuda()` - always use `device` parameter
- Use `.to(device)` consistently for tensor operations
- Check device compatibility before operations

### Checkpoint Format
- Checkpoints must include: model state, optimizer state, config, iter_num
- Use descriptive checkpoint names with timestamps
- Always validate checkpoint compatibility on load

### Code Style
- Follow GPT-2 paper conventions for naming (c_attn, c_proj, etc.)
- Use descriptive variable names for hyperparameters
- Keep architectural choices commented with paper references
