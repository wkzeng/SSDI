# ssdi-gen

SSDI-based federated data partition generator.

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

```
