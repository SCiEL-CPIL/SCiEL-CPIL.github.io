"""Optimal market making with inventory risk.

The objective and dynamics do not depend on t (mid is an exogenous random walk
that never enters the objective, and there is no terminal charge). Hence the
exact optimal policy is static: bid/ask depth depend only on (inventory, regime).
We precompute the exact optimal policy once per regime by backward induction over
the finite inventory state space, then look up depths at every step.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 300.0 - 12.0  # safety margin inside the 300 s wall-clock limit

# depth grid: distances below/above mid. Values below -3/k are invalid, so keep
# every candidate strictly above -3/k (the 3/k cap is never binding for optimal
# depths, which are positive).
_KMIN = 0.75
_DMIN = -3.0 / _KMIN + 1e-9
_DEPTH = np.arange(_DMIN, 0.2, 0.01)


def _build_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = 2 * Q + 1
    q0 = Q
    cost = 0.5 * gamma * sigma * sigma * dt
    # state index i -> inventory q = i - Q
    qarr = np.arange(N) - Q
    c = cost * qarr * qarr
    nD = _DEPTH.shape[0]
    d = _DEPTH
    # fill probability for each depth level, and the depth itself
    pb = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
    pa = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
    db = d.copy()
    da = d.copy()
    V = np.zeros(N)
    for _ in range(steps):
        # bid action value for every (state, depth)
        bv = db[None, :] + V[1:, None]
        # ask action value
        av = da[None, :] + V[:-1, None]
        bv = bv * pb[None, :] + (1.0 - pb[None, :]) * V[1:, None]
        av = av * pa[None, :] + (1.0 - pa[None, :]) * V[:-1, None]
        bbest = bv.max(axis=1)
        abest = av.max(axis=1)
        nb = np.zeros(N)
        na = np.zeros(N)
        nb[1:] = bbest
        na[:-1] = abest
        both = nb + na
        onlyb = nb + V[1:]
        onlya = na + V[:-1]
        none = V
        cand = np.stack([both, onlyb, onlya, none], axis=1)
        act = cand.argmax(axis=1)
        V = cand[np.arange(N), act] + c
        # record chosen depths (store for the final iteration below)
        bi = np.full(N, -1)
        ai = np.full(N, -1)
        bi[1:] = bv.argmax(axis=1)
        ai[:-1] = av.argmax(axis=1)
    # chosen depth indices for the optimal (final) policy
    return bi, ai, d


_cache = {}


def _get_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    pol = _cache.get(key)
    if pol is None:
        pol = _build_policy(params["sigma"], params["A"], params["k"],
                            params["gamma"], params["T"], params["steps"],
                            params["Q"])
        _cache[key] = pol
    return pol


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths), float arrays of shape (len(inventory),)."""
    Q = params["Q"]
    bi, ai, d = _get_policy(params)
    q = np.asarray(inventory)
    idx = q + Q
    # clamp to valid range for safety
    idx = np.clip(idx, 0, 2 * Q)
    # bid: no bid fill at q = Q  ->  no bid quote
    # ask: no ask fill at q = -Q ->  no ask quote
    bid = d[bi[idx].astype(np.int64)]
    ask = d[ai[idx].astype(np.int64)]
    bid = bid.astype(np.float64)
    ask = ask.astype(np.float64)
    bid[q >= Q] = np.inf
    ask[q <= -Q] = np.inf
    return bid, ask
# EVOLVE-BLOCK-END
