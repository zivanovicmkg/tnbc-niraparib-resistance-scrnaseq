# Environments

Two conda environments were used. PyDESeq2 is kept separate because it pins
package versions that conflict with the scVelo/CellRank stack.

| File | Used for |
|---|---|
| `b2bio.yml` | everything except the GSE297522 validation |
| `deseq.yml` | `analysis/07_validation_GSE297522/` |

Exported with:

```bash
conda env export -n b2bio --no-builds | grep -v '^prefix:' > env/b2bio.yml
conda env export -n deseq --no-builds | grep -v '^prefix:' > env/deseq.yml
```
