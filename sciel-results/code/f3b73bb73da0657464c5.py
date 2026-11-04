"""Suite task: signed equal-width step functions minimizing the max-abs autoconvolution.

Design (from scratch):
  Phase A: continuous L2 relaxation via gradient descent on the smooth surrogate
           E = sum (autoconv)^2.  This guides the signal away from high-peak regions
           that pure discrete search gets stuck in.
  Phase B: project to a ±1 signal (sign + zero-median tie-break), then discrete
           single-flip polish that directly minimizes the true objective
           M = max |autoconv| / (sum)^2.
  A hard wall-clock deadline guard returns the best-so-far with a safety margin.
"""

import time

import numpy as np

N = 400
DEADLINE = 140.0  # wall-clock budget; evaluator kills at 150 s, keep a margin


def _auto(x):
    """Full linear autoconvolution of x (length 2N-1), via FFT (zero-padded)."""
    n = x.size
    m = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, m)
    return np.fft.irfft(X * np.conj(X), m)[:2 * n - 1]


def _flip_cost(x, i, ac):
    """Delta in M = max|ac| / S^2 if we flip bit i (x[i] -> -x[i]).
    S = sum(x) is even, so S^2 is invariant to a single flip."""
    s2 = float(np.sum(x) ** 2)
    if s2 == 0.0:
        return 0.0
    d = -2.0 * x[i]
    n = x.size
    # new autoconvolution: ac' = ac + d * (c + flip(c))
    c = _auto(x)
    base = ac + d * c
    # add the reflected (flipped) contribution
    idx = np.arange(2 * n - 1)
    r = 2 * (n - 1) - idx
    base = base + d * c[r]
    return float(np.max(np.abs(base))) / s2


def _polish(x, budget):
    """Iterated single-flip polish minimizing M = max|ac| / S^2."""
    ac = _auto(x)
    s2 = float(np.sum(x) ** 2)
    best = float(np.max(np.abs(ac))) / s2
    best_x = x.copy()
    n = x.size
    rng = np.random.default_rng(12345)
    order = rng.permutation(n)
    k = 0
    while time.time() < budget:
        improved = False
        for i in order:
            if time.time() >= budget:
                break
            if np.sum(x) == 0:
                # S^2 = 0 is a degenerate objective; nudge away with a cheap flip
                x[i] = -x[i]
                ac = _auto(x)
                s2 = float(np.sum(x) ** 2)
                if s2 > 0:
                    best = float(np.max(np.abs(ac))) / s2
                    best_x = x.copy()
                improved = True
                continue
            c = _flip_cost(x, i, ac)
            if c < best - 1e-12:
                x[i] = -x[i]
                ac = _auto(x)
                s2 = float(np.sum(x) ** 2)
                best = c
                best_x = x.copy()
                improved = True
        if not improved:
            break
    return best_x, best


def construct():
    t0 = time.time()
    rng = np.random.default_rng(7)

    # ---- Phase A: continuous L2 relaxation ----
    x = rng.standard_normal(N)
    best_x = np.sign(x)
    best_x[best_x == 0] = 1.0
    best_val = float('inf')

    lr = 0.5
    it = 0
    while time.time() < DEADLINE - 40.0:
        ac = _auto(x)
        # dE/dx[i] = 4 * x * auto(x)
        g = 4.0 * x * ac
        x = x - lr * g
        # periodic rescale to keep magnitudes healthy
        x = x / (np.linalg.norm(x) + 1e-12)

        # project and polish periodically to track the discrete objective
        it += 1
        if it % 25 == 0 or time.time() > DEADLINE - 40.0:
            b = np.sign(x)
            b[b == 0] = 1.0
            if np.sum(b) == 0:
                # fix a zero-sum degenerate signal
                b[np.argmax(np.abs(b))] = 1.0
            bx, bv = _polish(b, DEADLINE - 20.0)
            if bv < best_val:
                best_val = bv
                best_x = bx.copy()
        # gentle annealing of the learning rate
        lr = max(0.02, lr * 0.995)

    # ---- Final polish on the best continuous projection ----
    b = np.sign(x)
    b[b == 0] = 1.0
    if np.sum(b) == 0:
        b[np.argmax(np.abs(b))] = 1.0
    bx, bv = _polish(b, DEADLINE - 2.0)
    if bv < best_val:
        best_val = bv
        best_x = bx.copy()

    # safety: always return a valid ±1 signal of length N
    out = np.sign(best_x)
    out[out == 0] = 1.0
    return out
