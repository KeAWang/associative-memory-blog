# /// script
# dependencies = ["numpy"]
# ///
"""Crosstalk in a key-value memory, for the score functions in the post.

Stores M random unit keys and values in R^d, queries with a stored key, and
reports |crosstalk| / |signal|, where f(x) = sum_i v_i * score(k_i^T x).

Run: uv run crosstalk.py
Also writes data/crosstalk.json, the data behind the charts in the post
"Every layer is an associative memory" (https://keawang.github.io/posts/growing-memory-layers/).
"""
import json
import math
from pathlib import Path

import numpy as np

seeds = [0, 1, 2, 3, 4]  # the chart averages over these
d = 64
beta = math.sqrt(d)  # sharpness for the exponential scores, as in softmax attention
Ms = [64, 256, 1024, 4096]
chart_Ms = [2**p for p in range(4, 15)]  # 16 ... 16384


def unit(n, rng):
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


relu = lambda s: np.maximum(s, 0)
silu = lambda s: s / (1 + np.exp(-s))
# name -> (score function, predicted ratio from overlaps s ~ N(0, 1/d), or None)
scores = {
    "s": (lambda s: s, lambda M: math.sqrt(M / d)),
    "ReLU(s)": (relu, lambda M: math.sqrt(M / (2 * d))),
    "ReLU(s)*s": (lambda s: relu(s) * s, lambda M: math.sqrt(M * 3 / 2) / d),
    "ReLU(s)^3": (lambda s: relu(s) ** 3, lambda M: math.sqrt(M * 15 / 2) / d**1.5),
    "SiLU(s)*s": (lambda s: silu(s) * s, None),
    # the exponential and its truncated series, at sharpness beta
    "exp2": (lambda s: 1 + beta * s, None),
    "exp3": (lambda s: 1 + beta * s + (beta * s) ** 2 / 2, None),
    "exp4": (lambda s: 1 + beta * s + (beta * s) ** 2 / 2 + (beta * s) ** 3 / 6, None),
    "exp": (lambda s: np.exp(beta * s), None),
}

def ratios(M, seed=0):
    """Median |crosstalk| / |signal| over 200 stored-key queries, for each score."""
    rng = np.random.default_rng(seed)
    K, V = unit(M, rng), unit(M, rng)
    S = K[:200] @ K.T  # 200 stored keys as queries, scored against all M keys
    out = {}
    for name, (score, predict) in scores.items():
        W = np.array(score(S))  # copy: the identity score would alias S
        signal = np.diag(W[:, :200]).copy()
        np.fill_diagonal(W[:, :200], 0)
        ratio = float(np.median(np.linalg.norm(W @ V, axis=1) / signal))
        if predict and M >= 64:  # sanity check vs the Gaussian (RMS) estimate; too few terms below 64
            # 30%: the median sits below the RMS when a few large terms dominate (ReLU^3 at small M)
            assert abs(ratio / predict(M) - 1) < 0.3, (name, M, ratio, predict(M))
        out[name] = ratio
    return out


print(f"d = {d}: |crosstalk| / |signal| when querying a stored key")
print("M".rjust(6) + "".join(name.rjust(11) for name in scores))  # columns follow `scores`
for M in Ms:
    print(str(M).rjust(6) + "".join(f"{r:11.2f}" for r in ratios(M).values()))

chart = {"d": d, "beta": beta, "M": chart_Ms, "series": {name: [] for name in ["s", "ReLU(s)", "ReLU(s)*s", "ReLU(s)^3", "exp2", "exp3", "exp4", "exp"]}}
chart["seeds"] = seeds
for M in chart_Ms:
    runs = [ratios(M, seed) for seed in seeds]
    for name in chart["series"]:
        chart["series"][name].append(round(float(np.mean([r[name] for r in runs])), 4))
out = Path(__file__).resolve().parent / "data" / "crosstalk.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(chart, indent=1))
print(f"wrote {out}")

# The optimal (least-squares) linear memory: querying stored key j weights the values by
# column j of P = K (K^T K)^{-1} K^T. Exact (zero noise) when M <= d, so only M > d is plotted.
def lsq_ratio(M, seed, queries=200):
    if M <= d:
        return 0.0
    rng = np.random.default_rng(seed)
    K, V = unit(M, rng), unit(M, rng)
    P = K[:queries] @ np.linalg.solve(K.T @ K, K.T)  # rows j of P, for the first `queries` keys
    signal = np.diag(P[:, :queries]).copy()
    np.fill_diagonal(P[:, :queries], 0)
    return float(np.median(np.linalg.norm(P @ V, axis=1) / signal))


# Its own, denser grid just past d, where the noise rises from zero like sqrt(M/d - 1)
lsq_Ms = [65, 66, 68, 72, 80, 96, 112] + [M for M in chart_Ms if M >= 128]
chart["lsq"] = {"M": lsq_Ms, "v": [round(float(np.mean([lsq_ratio(M, seed) for seed in seeds])), 4) for M in lsq_Ms]}
print("least squares:", list(zip(lsq_Ms, chart["lsq"]["v"])))
print("sqrt(M/d - 1):", [round(math.sqrt(M / d - 1), 4) for M in lsq_Ms])

# Long contexts: softmax attention's score, with a fixed sharpness and with log-length scaling,
# out to about 2 million pairs. Float32 and 50 queries per seed keep this within memory.
M0 = 1024  # plays Qwen's training length: beta = sqrt(d) up to here, then times log M / log M0 (LogN-Scaling)
long_Ms = [2**p for p in range(4, 22)]  # 16 ... 2,097,152


def long_ratio(M, beta_M, seed, queries=50):
    rng = np.random.default_rng(seed)
    K, V = unit(M, rng).astype(np.float32), unit(M, rng).astype(np.float32)
    S = K[:queries] @ K.T
    W = np.exp(beta_M * (S - 1))  # dividing every weight by e^beta leaves the ratio unchanged
    signal = np.diag(W[:, :queries]).copy()
    np.fill_diagonal(W[:, :queries], 0)
    return float(np.median(np.linalg.norm(W @ V, axis=1) / signal))


# Long-context rules for the sharpness, all with the same reference length M0:
rules = {
    "fixed": lambda M: beta,                                            # plain softmax attention
    "logn": lambda M: beta * max(1.0, math.log(M) / math.log(M0)),      # Qwen LogN-Scaling (beyond the training length)
    "su": lambda M: beta * math.log(M) / math.log(M0),                  # Su (2021) entropy invariance; SSMax with fixed s
    "yarn": lambda M: beta * max(1.0, 0.1 * math.log(M / M0) + 1) ** 2,  # YaRN temperature (Qwen2.5-1M)
}
long = {"M0": M0, "M": long_Ms, "series": {k: [] for k in rules}}
for M in long_Ms:
    for key, rule in rules.items():
        long["series"][key].append(round(float(np.mean([long_ratio(M, rule(M), seed) for seed in seeds])), 6))
    print(f"M={M:8d}  " + "  ".join(f"{k}={long['series'][k][-1]:.5f}" for k in rules), flush=True)

fixed = long["series"]["fixed"]
cross = next((i for i, r in enumerate(fixed) if r >= 1), None)
if cross:  # interpolate the crossing in log-log
    lo, hi = cross - 1, cross
    t = -math.log(fixed[lo]) / (math.log(fixed[hi]) - math.log(fixed[lo]))
    long["fixed_crossing"] = round(math.exp(math.log(long_Ms[lo]) + t * (math.log(long_Ms[hi]) - math.log(long_Ms[lo]))))
    print("fixed-beta crossing near M =", long["fixed_crossing"])
chart["long"] = long
out.write_text(json.dumps(chart, indent=1))
print(f"wrote {out}")
