"""Model-and-optimize solver for no-dividing-3-sum task.
Parameterize the selected set by value bands / parametric families,
and optimize the parameters against the divisibility objective using
fast precomputed feasibility checks.
"""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

# Precompute divisible sums for fast lookup
_divisible_sums = {}
for x in range(1, _N + 1):
    _divisible_sums[x] = set(range(x, 574, x))

def _is_valid_addition_fast(S_set, x):
    """Check if adding x to valid set S_set keeps it valid."""
    S_list = list(S_set)
    n = len(S_list)
    
    if n == 0:
        return True
    
    # Check 1: x does not divide sum of any 3 in S_set
    div_sums = _divisible_sums[x]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = S_list[i] + S_list[j] + S_list[k]
                if s in div_sums:
                    return False
    
    # Check 2: for each a in S_set, a does not divide sum of x + 2 others from S_set
    for i, a in enumerate(S_list):
        div_sums_a = _divisible_sums[a]
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + S_list[j] + S_list[k]
                if s in div_sums_a:
                    return False
    
    return True

def _greedy_optimized(order):
    """Greedy with given order."""
    S = set()
    for x in order:
        if x in S:
            continue
        if _is_valid_addition_fast(S, x):
            S.add(x)
            if len(S) >= _MAX:
                break
    return S

def _local_search_model(S_init, time_budget=20.0):
    """Local search with insertions, 1-out-2-in, and swaps."""
    S = set(S_init)
    start_time = time.time()
    best = set(S)
    best_len = len(S)
    all_vals = list(range(1, _N + 1))
    no_improve_count = 0
    
    while time.time() - start_time < time_budget and _time_left() > 5:
        improved = False
        
        # Try insertions
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            for x in candidates[:80]:
                if _is_valid_addition_fast(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = set(S)
                    break
        
        # Try 1-out-2-in moves (net gain +1)
        if len(S) < _MAX and len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:15]:
                S_temp = S - {a}
                # Try to add 2 elements to S_temp
                for x in candidates[:30]:
                    if _is_valid_addition_fast(S_temp, x):
                        S_temp2 = S_temp | {x}
                        for y in candidates[:30]:
                            if y == x: continue
                            if _is_valid_addition_fast(S_temp2, y):
                                S = S_temp2 | {y}
                                improved = True
                                if len(S) > best_len:
                                    best_len = len(S)
                                    best = set(S)
                                break
                        if improved:
                            break
                if improved:
                    break
        
        # Try swaps (1 out, 1 in)
        if len(S) >= 1 and not improved:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S]
            random.shuffle(candidates)
            
            for a in to_remove[:25]:
                for x in candidates[:25]:
                    S_new = S - {a}
                    if x not in S_new and _is_valid_addition_fast(S_new, x):
                        S = S_new
                        S.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = set(S)
                        break
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            # Perturb: remove random elements based on stagnation
            if len(S) > 2:
                if no_improve_count <= 2:
                    to_remove = random.sample(list(S), 2)
                elif no_improve_count <= 5:
                    to_remove = random.sample(list(S), 3)
                else:
                    to_remove = random.sample(list(S), 4)
                S -= set(to_remove)
                no_improve_count = 0
    
    return best

def _band_optimization():
    """Optimize band parameters."""
    best = set()
    best_len = 0
    
    configs = [
        [(100, 192)],
        [(80, 192)],
        [(60, 192)],
        [(50, 192)],
        [(40, 192)],
        [(100, 192), (20, 39)],
        [(100, 192), (30, 49)],
        [(90, 192), (20, 50)],
        [(80, 192), (15, 40)],
        [(70, 192), (10, 35)],
        [(120, 192), (40, 70)],
        [(110, 192), (30, 60)],
        [(100, 192), (25, 50)],
        [(95, 192), (20, 45)],
        [(90, 192), (15, 40)],
        [(110, 192), (10, 30)],
        [(105, 192), (15, 45)],
        [(95, 192), (10, 40)],
    ]
    
    for config in configs:
        for _ in range(8):
            S = _model_parameterize(config)
            if len(S) > best_len:
                best_len = len(S)
                best = set(S)
    
    return best

def _model_parameterize(band_widths):
    """Parameterize the set by value bands. 
    band_widths: list of (start, end) ranges to sample from.
    Returns a candidate set."""
    S = set()
    for start, end in band_widths:
        # Pick values from [start, end]
        candidates = list(range(start, min(end, _N) + 1))
        random.shuffle(candidates)
        for x in candidates:
            if len(S) >= _MAX:
                break
            if x not in S and _is_valid_addition_fast(S, x):
                S.add(x)
    return S

def construct():
    """Main entry point: find maximum valid set."""
    random.seed(42)  # Deterministic
    
    # Phase 1: Band optimization (model-and-optimize)
    best = _band_optimization()
    
    # Phase 2: Greedy with various orders
    if _time_left() > 30:
        orders = [
            range(_N, 0, -1),
            range(1, _N + 1),
            list(range(50, _N + 1)) + list(range(1, 50)),
            list(range(80, _N + 1)) + list(range(1, 80)),
            list(range(100, _N + 1)) + list(range(1, 100)),
            list(range(120, _N + 1)) + list(range(1, 120)),
        ]
        for order in orders:
            if _time_left() < 30:
                break
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = S
    
    # Phase 3: Multi-start randomized greedy
    if _time_left() > 20:
        for _ in range(30):
            if _time_left() < 20:
                break
            order = list(range(1, _N + 1))
            weights = [random.random() * 50 + v for v in order]
            order.sort(key=lambda v: -weights[v - 1])
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = S
    
    # Phase 4: Local search from best
    if _time_left() > 10:
        budget = min(25.0, _time_left() - 5)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = S
    
    # Phase 5: Another local search round if time permits
    if _time_left() > 5:
        budget = min(8.0, _time_left() - 2)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = S
    
    return list(best)
# EVOLVE-BLOCK-END
