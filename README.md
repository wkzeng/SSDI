# ssdi-gen

SSDI-based federated data partition generator.

SSDI ranges from 0 to 1, where `0` denotes an IID partition and larger values indicate stronger heterogeneity.

## Install

```bash
pip install -e .
```

## Recommended imports

```python
from ssdi_gen import generate_ssdi_matrix_structured
```

## Single generation

```python
from ssdi_gen import generate_ssdi_matrix

df = generate_ssdi_matrix_structured(
    label=10,    
    client=30,    
    datasize=60000,
    ssdi=0.3,
    structure_mode=[ "mixed" ],   # "skew" | "mixed" | "coverage"
    )

`structure_mode` controls the heterogeneity pattern:
- `"skew"`: more LDS-dominant
- `"mixed"`: balanced between LDS and LCD
- `"coverage"`: more LCD-dominant

It can also be specified as a continuous value in `[-1, 1]`, where values closer to `-1` indicate more skew/LDS-dominant patterns, values near `0` indicate mixed patterns, and values closer to `1` indicate more coverage/LCD-dominant patterns.
```
