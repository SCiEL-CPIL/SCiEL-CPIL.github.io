# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 10
_DEMANDS = [
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
]
_D = 15

# Edge e joins node e and (e+1) mod 10. Clockwise from s to t uses edges s, s+1, ..., t-1.
# Clockwise edge e is traversed iff e is in [s, t) clockwise, i.e. (e - s) mod N < (t - s) mod N.
_EDGE = np.arange(_N)
_CCW_MASK = np.zeros((_D, _N), dtype=np.float64)
_CCW_WEIGHT = np.zeros(_D, dtype=np.float64)
for _i, (_s, _t, _w) in enumerate(_DEMANDS):
    for _e in range(_N):
        if (_e - _s) % _N < (_t - _s) % _N:
            _CCW_MASK[_i, _e] = 1.0
    _CCW_WEIGHT[_i] = _w
_CCW_BASE = _CCW_MASK @ _CCW_WEIGHT          # load on each edge if all routed clockwise
_CCW_EDGE = _CCW_MASK.T @ _CCW_WEIGHT       # per-demand weight vector per edge (for flips)


def _load_from_bits(bits):
    """bits: (D,) with 1 = counter-clockwise. Returns (N,) edge loads."""
    return _CCW_BASE + _CCW_EDGE @ bits


def _max_load(bits):
    return float(np.max(_load_from_bits(bits)))


def _local_search(bits, iters, deadline):
    """Hill-climb: try each single flip, keep improving; stop on deadline."""
    bits = bits.copy()
    best = _max_load(bits)
    for _ in range(iters):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(_D):
            b2 = bits.copy()
            b2[i] = 1.0 - b2[i]
            m = _max_load(b2)
            if m < best - 1e-9:
                best = m
                bits = b2
                improved = True
        if not improved:
            break
    return bits, best


def _anneal(bits, t0, t1, iters, deadline):
    """Simulated annealing on top of a local-search result."""
    bits = bits.copy()
    best = _max_load(bits)
    best_bits = bits.copy()
    cur = best
    for k in range(iters):
        if time.time() >= deadline:
            break
        T = t0 * (t1 / t0) ** (k / max(1, iters - 1))
        i = int(np.random.randint(_D))
        b2 = bits.copy()
        b2[i] = 1.0 - b2[i]
        m = _max_load(b2)
        if m < cur or np.random.rand() < np.exp(-(m - cur) / max(T, 1e-9)):
            bits = b2
            cur = m
            if m < best - 1e-9:
                best = m
                best_bits = bits.copy()
    return best_bits, best


def ring_routing() -> np.ndarray:
    deadline = time.time() + 600.0 - 20.0  # safety margin
    rng = np.random.default_rng(12345)

    # ---- Phase 1: strong greedy initial solution ----
    # Start all-clockwise, then flip demands to reduce the max load.
    bits = np.zeros(_D)
    # Greedy: repeatedly flip the demand whose flip most reduces the current max edge load.
    for _ in range(200):
        if time.time() >= deadline:
            break
        cur_max = _max_load(bits)
        best_gain = 1e-9
        best_i = -1
        for i in range(_D):
            b2 = bits.copy()
            b2[i] = 1.0 - b2[i]
            m = _max_load(b2)
            gain = cur_max - m
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        bits[best_i] = 1.0 - bits[best_i]

    # Local search to polish
    bits, best = _local_search(bits, 50, deadline)
    best_bits = bits.copy()

    # ---- Phase 2: simulated annealing from the greedy solution ----
    bits, best = _anneal(bits, t0=2.0, t1=1e-4, iters=20000, deadline=deadline)
    if best < best_bits.max() if False else True:
        pass
    if best < _max_load(best_bits):
        best_bits = bits.copy()
        best = _max_load(best_bits)

    # ---- Phase 3: random restarts with local search + short anneal ----
    remaining_restarts = 0
    while time.time() < deadline:
        remaining_restarts += 1
        rb = (rng.random(_D) < 0.5).astype(np.float64)
        rb, mb = _local_search(rb, 40, deadline)
        if time.time() < deadline:
            rb, mb = _anneal(rb, t0=1.5, t1=1e-4, iters=5000, deadline=deadline)
        if mb < best - 1e-9:
            best = mb
            best_bits = rb.copy()

    # Final polish
    best_bits, best = _local_search(best_bits, 50, deadline)

    out = np.zeros(_D)
    out[best_bits > 0.5] = 1.0
    return out


# EVOLVE-BLOCK-END
