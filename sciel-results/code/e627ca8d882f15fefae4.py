"""Low discrepancy on all arithmetic progressions - n=256."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
_DEADLINE = 75.0  # safety margin before 90s kill
_t0 = time.time()

# Precompute all arithmetic progressions (start, step, length>=1) within [1,256].
# Represent each AP as a tuple of 0-based indices.
_APS = []
for start in range(N):
    for step in range(1, N):
        # collect indices start, start+step, ... < N
        seq = list(range(start, N, step))
        if len(seq) >= 1:
            _APS.append(seq)

# For speed: keep only APs of length >= 2 as "real" constraints (singletons always sum to +/-1).
_APS_L2 = [a for a in _APS if len(a) >= 2]
# Group by length for faster incremental updates is complex; instead keep list of index-lists.
# Precompute, for each position, the list of APs that contain it (for incremental update).
_pos_aps = [[] for _ in range(N)]
for _i, _seq in enumerate(_APS_L2):
    for _idx in _seq:
        _pos_aps[_idx].append(_i)

# Precompute length of each AP.
_APS_LEN = [len(a) for a in _APS_L2]

def _full_eval(signs):
    """Return max absolute signed sum over all APs (length>=2) and singletons."""
    mx = 1  # singleton bound
    for seq in _APS_L2:
        s = 0
        for idx in seq:
            s += signs[idx]
        a = s if s >= 0 else -s
        if a > mx:
            mx = a
    return mx

def _eval_partial(signs):
    """Fast full eval used during search."""
    return _full_eval(signs)

def _construct():
    global _t0
    rng = random.Random(20240517)

    # Start from a good initial configuration: alternate pattern is low for small steps.
    # Use a few candidate starts and pick the best.
    candidates = []
    # Pattern 1: alternating
    cand1 = [1 if (i % 2 == 0) else -1 for i in range(N)]
    candidates.append(cand1)
    # Pattern 2: Thue-Morse-like (parity of bit count)
    def thue_morse(n):
        c = 0
        x = n
        while x:
            c ^= (x & 1)
            x >>= 1
        return 1 if c == 0 else -1
    cand2 = [thue_morse(i) for i in range(N)]
    candidates.append(cand2)
    # Pattern 3: random
    cand3 = [rng.choice((-1, 1)) for _ in range(N)]
    candidates.append(cand3)
    # Pattern 4: period-4  + + - -
    cand4 = [1, 1, -1, -1] * (N // 4)
    candidates.append(cand4)

    best = None
    best_val = 10**9
    for cand in candidates:
        v = _full_eval(cand)
        if v < best_val:
            best_val = v
            best = cand[:]

    # Local search: hill-climbing with random restarts, deadline-guarded.
    def try_improve(signs, val):
        # Greedy: for each position, try flipping; accept if improves max AP sum.
        # To be fast, only consider flipping positions that participate in near-max APs.
        # Simpler: random order, accept first improving flip (steepest-ish).
        order = list(range(N))
        rng.shuffle(order)
        improved = True
        while improved:
            improved = False
            for pos in order:
                if time.time() - _t0 > _DEADLINE:
                    return signs, val
                old = signs[pos]
                signs[pos] = -old
                # Recompute affected APs only (incremental)
                # We'll just do a full eval but that's O(#APs * len). #APs ~ large.
                # Instead incremental: recompute sums for APs containing pos.
                new_val = _incremental_after_flip(signs, pos, val, old)
                if new_val < val:
                    val = new_val
                    improved = True
                    order = list(range(N))
                    rng.shuffle(order)
                else:
                    signs[pos] = old
        return signs, val

    # Incremental evaluation: maintain running sums per AP? Too heavy to keep in closure.
    # Instead, implement a cached-sum approach.
    # For simplicity and correctness within budget, use full eval but limit flips.
    # We'll do a bounded number of full-eval hill-climb iterations.
    def hill_climb(signs, val):
        # Each iteration: try a batch of random flips, keep improving.
        iters = 0
        while time.time() - _t0 < _DEADLINE and iters < 4000:
            iters += 1
            pos = rng.randrange(N)
            old = signs[pos]
            signs[pos] = -old
            nv = _full_eval(signs)
            if nv < val:
                val = nv
            else:
                signs[pos] = old
            # occasional 2-flip
            if iters % 5 == 0:
                p1 = rng.randrange(N); p2 = rng.randrange(N)
                if p1 == p2: continue
                o1 = signs[p1]; o2 = signs[p2]
                signs[p1] = -o1; signs[p2] = -o2
                nv = _full_eval(signs)
                if nv < val:
                    val = nv
                else:
                    signs[p1] = o1; signs[p2] = o2
        return signs, val

    signs, val = hill_climb(best[:], best_val)

    # A few random restarts if time permits.
    restarts = 0
    while time.time() - _t0 < _DEADLINE - 5 and restarts < 6:
        restarts += 1
        cand = [rng.choice((-1, 1)) for _ in range(N)]
        cv = _full_eval(cand)
        if cv < val:
            signs, val = hill_climb(cand, cv)

    return signs

def construct():
    global _t0
    _t0 = time.time()
    return _construct()
# EVOLVE-BLOCK-END
