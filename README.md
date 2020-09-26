## Installation

`pip install -U ukro_g2p`

## Example inference

```python
from ukro_g2p.predict import G2P

g2p = G2P('ukro-base-uncased')
pron = g2p(script_args.word)
print(f"{' '.join(pron)}")
```
