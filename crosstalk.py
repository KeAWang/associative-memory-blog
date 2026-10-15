# /// script
# dependencies = ["numpy"]
# ///
"""Interference in a key-value memory, for the score functions in the post.

Stores M random unit keys and values in R^d and queries the memory with x = rho * k_j + sqrt(1 - rho^2) * u, where k_j
is a stored key, u is a random unit vector perpendicular to it, and rho is the query's similarity to k_j (rho = 1 queries
with the stored key itself). Reports |noise| / |signal|, where f(x) = sum_i v_i w_i, the signal is v_j w_j and the noise
is every other term. Each curve is computed at every rho in `rhos`, averaged over `seeds`.

Run: uv run crosstalk.py
Writes data/crosstalk.json, the data behind the charts in the post
"Every layer is an associative memory" (https://keawang.github.io/posts/growing-memory-layers/).
"""
import json
import math
from pathlib import Path

import numpy as np

seeds = [0, 1, 2, 3, 4]
d = 64
beta = math.sqrt(d)  # sharpness for the exponential scores, as in softmax attention
rhos = [1.0]  # the query is the stored key itself; lower values mix in a random perpendicular direction
FAIL = 1e6  # stands in for "no signal" (e.g. a ReLU with rho <= 0); the charts pin it to the top
chart_Ms = [2**p for p in range(4, 15)]  # 16 ... 16384
lsq_Ms = [16, 32, 64, 65, 66, 68, 72, 80, 96, 112] + [M for M in chart_Ms if M >= 128]
M0 = 1024  # plays Qwen's training length for the long-context rules
long_Ms = [2**p for p in range(4, 22)]  # 16 ... 2,097,152


def unit(n, rng, dtype=np.float64):
    x = rng.standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(dtype)


def perp_units(K, rng):
    """For each row k of K, a random unit vector perpendicular to k."""
    g = rng.standard_normal(K.shape).astype(K.dtype)
    g -= np.sum(g * K, axis=1, keepdims=True) * K
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def ratio(W, V):
    """Median |noise| / |signal| over queries; row q of W holds the weights for a query near stored key q."""
    Q = W.shape[0]
    signal = np.abs(np.diag(W[:, :Q])).copy()
    W = W.copy()
    np.fill_diagonal(W[:, :Q], 0)
    noise = np.linalg.norm(W @ V, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(signal > 0, noise / np.maximum(signal, 1e-300), np.inf)
    return float(np.median(r))


relu = lambda s: np.maximum(s, 0)
scores = {
    "s": lambda s: s,
    "ReLU(s)": relu,
    "ReLU(s)*s": lambda s: relu(s) * s,
    "ReLU(s)^3": lambda s: relu(s) ** 3,
    # the exponential and its series truncated after 2, 3 and 4 terms
    "exp2": lambda s: 1 + beta * s,
    "exp3": lambda s: 1 + beta * s + (beta * s) ** 2 / 2,
    "exp4": lambda s: 1 + beta * s + (beta * s) ** 2 / 2 + (beta * s) ** 3 / 6,
    "exp": lambda s: np.exp(beta * (s - 1)),  # e^{beta s} / e^beta: same ratios, no overflow
}
# rho = 1 sanity checks against the Gaussian (RMS) estimates; 30% because the median sits below the RMS
# when a few large terms dominate (ReLU^3 at small M), and only for M >= 64 (too few terms below)
predict = {
    "s": lambda M: math.sqrt(M / d),
    "ReLU(s)": lambda M: math.sqrt(M / (2 * d)),
    "ReLU(s)*s": lambda M: math.sqrt(M * 3 / 2) / d,
    "ReLU(s)^3": lambda M: math.sqrt(M * 15 / 2) / d**1.5,
}


def draw(M, seed, Q, dtype=np.float64):
    """Keys, values, and the similarities of Q stored-key queries and Q perpendicular directions to every key."""
    rng = np.random.default_rng(seed)
    K, V = unit(M, rng, dtype), unit(M, rng, dtype)
    U = perp_units(K[:Q], rng)
    return K, V, U, K[:Q] @ K.T, U @ K.T


def mixed(Sk, Su, rho):
    return rho * Sk + math.sqrt(max(0.0, 1 - rho**2)) * Su


def clean(x):
    return FAIL if not math.isfinite(x) else round(x, 6)


chart = {"d": d, "beta": beta, "seeds": seeds, "rhos": rhos, "M": chart_Ms,
         "series": {k: [[] for _ in rhos] for k in scores}}

# Powers and exponentials, with M up to 16384
for M in chart_Ms:
    runs = {k: [[] for _ in rhos] for k in scores}
    for seed in seeds:
        K, V, U, Sk, Su = draw(M, seed, min(200, M))
        for ri, rho in enumerate(rhos):
            S = mixed(Sk, Su, rho)
            for k, f in scores.items():
                runs[k][ri].append(ratio(np.array(f(S)), V))
    for k in scores:
        for ri in range(len(rhos)):
            chart["series"][k][ri].append(clean(np.mean(runs[k][ri])))
    one = rhos.index(1.0)
    if M >= 64:
        for k, p in predict.items():
            r = chart["series"][k][one][-1]
            assert abs(r / p(M) - 1) < 0.3, (k, M, r, p(M))
    print(f"M={M:6d}  rho=1: " + "  ".join(f"{k}={chart['series'][k][one][-1]:.3g}" for k in scores), flush=True)

# The optimal (least-squares) linear memory, W = V^T (K^T)^+: the weights for a query x are (K^T)^+ x.
# Exact (zero noise) for M <= d when the query is a stored key; its own grid is denser just past d.
chart["lsq"] = {"M": lsq_Ms, "v": [[] for _ in rhos]}
for M in lsq_Ms:
    runs = [[] for _ in rhos]
    for seed in seeds:
        K, V, U, _, _ = draw(M, seed, min(200, M))
        Kp = np.linalg.pinv(K.T)  # M x d
        Pk, Pu = (Kp @ K[:min(200, M)].T).T, (Kp @ U.T).T  # weights for the stored keys and the perpendicular parts
        for ri, rho in enumerate(rhos):
            r = ratio(mixed(Pk, Pu, rho), V)
            runs[ri].append(0.0 if r < 1e-9 else r)
    for ri in range(len(rhos)):
        chart["lsq"]["v"][ri].append(clean(np.mean(runs[ri])))
print("least squares, rho=1:", list(zip(lsq_Ms, chart["lsq"]["v"][rhos.index(1.0)])), flush=True)

# Long contexts: softmax attention out to about 2 million pairs, under four rules for the sharpness, all with the
# same reference length M0. Float32 and fewer queries at large M keep this within memory.
rules = {
    "fixed": lambda _: beta,                                             # plain softmax attention
    "logn": lambda M: beta * max(1.0, math.log(M) / math.log(M0)),       # Qwen LogN-Scaling (beyond the training length)
    "su": lambda M: beta * math.log(M) / math.log(M0),                   # Su (2021) entropy invariance; SSMax with fixed s
    "yarn": lambda M: beta * max(1.0, 0.1 * math.log(M / M0) + 1) ** 2,  # YaRN temperature (Qwen2.5-1M)
}
chart["long"] = {"M0": M0, "M": long_Ms, "series": {k: [[] for _ in rhos] for k in rules}}
for M in long_Ms:
    runs = {k: [[] for _ in rhos] for k in rules}
    for seed in seeds:
        K, V, U, Sk, Su = draw(M, seed, max(16, min(50, 2**25 // M)), np.float32)
        for ri, rho in enumerate(rhos):
            S = mixed(Sk, Su, rho)
            for k, rule in rules.items():
                runs[k][ri].append(ratio(np.exp(rule(M) * (S - 1)), V))
        del K, V, U, Sk, Su
    for k in rules:
        for ri in range(len(rhos)):
            chart["long"]["series"][k][ri].append(clean(np.mean(runs[k][ri])))
    one = rhos.index(1.0)
    print(f"M={M:8d}  rho=1: " + "  ".join(f"{k}={chart['long']['series'][k][one][-1]:.3g}" for k in rules), flush=True)

out = Path(__file__).resolve().parent / "data" / "crosstalk.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(chart, separators=(",", ":")))
print(f"wrote {out}")
