# Results

## Simple

Size: 2000
Runs: 10

| Library | Mode   | Time  | Notes                                       |
| ------- | ------ | ----- | ------------------------------------------- |
| `numpy` | n/a    | 1.70  |                                             |
| `torch` | `cpu`  | 2.02  | FIXME slower than `numpy`                   |
| `torch` | `cuda` | 20.20 | FIXME slower than `numpy` and `torch` `cpu` |
