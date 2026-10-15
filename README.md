# associative-memory-blog

Code for the experiments in the blog post
[Every layer is an associative memory](https://keawang.github.io/posts/growing-memory-layers/).

`crosstalk.py` stores `M` random unit key-value pairs in `d = 64` dimensions, queries the memory
with a stored key, and measures how much of the retrieved vector is interference from the other
memories (the noise-to-signal ratio). It compares:

- a linear (outer-product) memory and the optimal least-squares linear memory, which is exact up to `M = d`,
- an MLP-style nonlinearity on each similarity: ReLU, ReLU², ReLU³,
- the exponential `e^{βs}` (softmax attention) and its series truncated after 2, 3 and 4 terms,
- softmax attention out to about 2 million pairs, with a fixed sharpness and with the
  log-length scalings used by Qwen (LogN-Scaling), Su (2021) / Scalable-Softmax, and YaRN.

Results are averaged over 5 random seeds and written to `data/crosstalk.json`.

## Run

```bash
uv run crosstalk.py
```

`uv` reads the script's inline dependencies (just `numpy`). The full run takes under a minute.
