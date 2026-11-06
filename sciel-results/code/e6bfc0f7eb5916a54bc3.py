"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0  # seconds; return with margin


def _triples(n):
    """List of triples (a,b,c), a<b<c, in lexicographic order."""
    return list(combinations(range(n), 3))


def _triples_to_idx(n):
    """Map (a,b,c) -> index in lexicographic triple list."""
    triples = _triples(n)
    m = len(triples)
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx, m


def _build_quads(n, triples):
    """For each 4-set, list of its 4 triple indices. Returns list of lists."""
    m = len(triples)
    # Precompute triple index via formula for speed
    def t_idx(a, b, c):
        # lexicographic index of (a,b,c) among triples of n
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
        quads.append(idxs)
    return quads


def _build_pents(n, triples):
    """For each 5-set, list of its 10 triple indices."""
    def t_idx(a, b, c):
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt
    pents = []
    for p in combinations(range(n), 5):
        a, b, c, d, e = p
        idxs = []
        for t in combinations(p, 3):
            idxs.append(t_idx(*t))
        pents.append(idxs)
    return pents


def _count_violations(colors, quads, pents):
    """Count red K4 (all 4 red) and blue K5 (all 10 blue)."""
    red_k4 = 0
    for idxs in quads:
        s = 0
        for i in idxs:
            s += colors[i] == 0
        if s == 4:
            red_k4 += 1
    blue_k5 = 0
    for idxs in pents:
        s = 0
        for i in idxs:
            s += colors[i] == 1
        if s == 10:
            blue_k5 += 1
    return red_k4, blue_k5


def _local_search(colors, quads, pents, time_budget):
    """Flip triples to reduce violations. Returns (colors, red_k4, blue_k5)."""
    colors = np.array(colors, dtype=np.int8)
    m = len(colors)
    
    # Build adjacency: for each triple, which quads and pents it belongs to
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)
    
    # Convert to numpy arrays for faster access
    tri_quads_arr = [np.array(tq, dtype=np.int32) if tq else np.array([], dtype=np.int32) for tq in tri_quads]
    tri_pents_arr = [np.array(tp, dtype=np.int32) if tp else np.array([], dtype=np.int32) for tp in tri_pents]
    
    # Maintain counts
    quad_red = np.zeros(len(quads), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        quad_red[qi] = np.sum(colors[idxs] == 0)
    
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    for pi, idxs in enumerate(pents):
        pent_blue[pi] = np.sum(colors[idxs] == 1)
    
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    total_v = red_k4 + blue_k5
    
    if total_v == 0:
        return colors.tolist(), red_k4, blue_k5

    t_end = time.time() + time_budget
    order = list(range(m))
    random.shuffle(order)
    improved = True
    stall_count = 0
    
    while improved and time.time() < t_end:
        improved = False
        stall_count = 0
        for t in order:
            if time.time() >= t_end:
                break
            cur = colors[t]
            new = 1 - cur
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            
            # Compute change in violations using vectorized ops
            tq = tri_quads_arr[t]
            tp = tri_pents_arr[t]
            
            if len(tq) > 0:
                old_q = quad_red[tq]
                new_q = old_q + delta_red
                dq = int(np.sum((new_q == 4) - (old_q == 4)))
            else:
                dq = 0
            
            if len(tp) > 0:
                old_p = pent_blue[tp]
                new_p = old_p + delta_blue
                dp = int(np.sum((new_p == 10) - (old_p == 10)))
            else:
                dp = 0
            
            if dq + dp < 0:
                # Apply flip
                colors[t] = new
                if len(tq) > 0:
                    quad_red[tq] += delta_red
                if len(tp) > 0:
                    pent_blue[tp] += delta_blue
                red_k4 += dq
                blue_k5 += dp
                improved = True
        
        # Reshuffle if stuck
        if not improved:
            stall_count += 1
            if stall_count >= 3:
                break
            random.shuffle(order)
    
    return colors.tolist(), red_k4, blue_k5


def _random_valid(n, time_budget):
    """Try to find a valid colouring on n vertices via random init + local search."""
    triples, idx, m = _triples_to_idx(n)
    quads = _build_quads(n, triples)
    pents = _build_pents(n, triples)
    
    # Multiple initialization strategies
    strategies = [
        lambda: (np.random.rand(m) < 0.35).astype(np.int8),
        lambda: (np.random.rand(m) < 0.40).astype(np.int8),
        lambda: (np.random.rand(m) < 0.30).astype(np.int8),
        lambda: (np.random.rand(m) < 0.45).astype(np.int8),
    ]
    
    # Also try structured initializations
    def structured_init():
        colors = np.zeros(m, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            # Various modular patterns
            colors[i] = (a + 2*b + 3*c) % 2
        return colors
    
    def structured_init2():
        colors = np.zeros(m, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            colors[i] = (a * b + b * c + c * a) % 2
        return colors
    
    strategies.append(structured_init)
    strategies.append(structured_init2)
    
    attempts = 0
    t_end = _START + time_budget
    
    while time.time() < t_end:
        attempts += 1
        # Cycle through strategies
        strat_idx = (attempts - 1) % len(strategies)
        colors = strategies[strat_idx]().tolist()
        
        # Time budget per attempt - decrease as we get closer to deadline
        remaining = t_end - time.time()
        if remaining < 2.0:
            break
        
        # Use more time for later attempts (they might be closer)
        ls_budget = min(3.0, remaining * 0.4)
        
        colors, r4, b5 = _local_search(colors, quads, pents, ls_budget)
        if r4 == 0 and b5 == 0:
            return colors
    
    return None


def construct():
    deadline = _START + _DEADLINE - 5.0  # safety margin
    best_n = 9
    best_colors = None
    
    # Seed with the known small valid construction
    from itertools import combinations as _c
    n0 = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in _c(range(n0), 3)]
    best_n = n0

    # Try to grow n from 10 upward, using time budget
    for n in range(10, 41):
        if time.time() >= deadline:
            break
        # Estimate time per n
        remaining = deadline - time.time()
        if n <= 20:
            budget = min(remaining, 20.0)
        elif n <= 28:
            budget = min(remaining, 30.0)
        else:
            budget = min(remaining, 40.0)
        if budget < 3.0:
            break
        colors = _random_valid(n, budget)
        if colors is not None:
            best_n = n
            best_colors = colors
        # If we found valid at n, try n+1 with remaining time
        if time.time() >= deadline:
            break

    return {'n': best_n, 'colors': best_colors}
