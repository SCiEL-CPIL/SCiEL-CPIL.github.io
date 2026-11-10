"""Optimized solver for no-dividing-3-sum task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

# Precomputation for fast validity checks
# divs[x][r] = list of values v in 1..192 such that v % x == r
_divs = [None] * (_N + 1)
# pairs_divs[a][r] = list of pairs (b, c) with b < c, b,c in 1..192, (b+c) % a == r
_pairs_divs = [None] * (_N + 1)

def _precompute():
    global _divs, _pairs_divs
    for x in range(1, _N + 1):
        _divs[x] = [[] for _ in range(x)]
        for v in range(1, _N + 1):
            _divs[x][v % x].append(v)
    
    for a in range(1, _N + 1):
        _pairs_divs[a] = [[] for _ in range(a)]
        # Generate all pairs (b, c) with b < c
        # Optimization: only store pairs that are valid candidates? 
        # We store all pairs, filter during check.
        for b in range(1, _N + 1):
            for c in range(b + 1, _N + 1):
                r = (b + c) % a
                _pairs_divs[a][r].append((b, c))

# Run precomputation once at start
_precompute()


def _is_valid_addition(S_set, x):
    """Check if adding x to valid set S_set keeps it valid.
    S_set is a set of integers.
    """
    # Check 1: x does not divide sum of any 3 in S
    # For any pair (b, c) in S, if there exists a in S such that a + b + c == 0 mod x, invalid.
    # a == -(b+c) mod x.
    for b, c in combinations(S_set, 2):
        r = (-(b + c)) % x
        # Check if any a in _divs[x][r] is in S_set and a != b and a != c
        for a in _divs[x][r]:
            if a in S_set and a != b and a != c:
                return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    # b + c == -x mod a.
    for a in S_set:
        r = (-x) % a
        # Check if any pair (b, c) in _pairs_divs[a][r] is in S_set and b,c != a
        for b, c in _pairs_divs[a][r]:
            if b in S_set and c in S_set and a != b and a != c:
                return False
                
    return True


def _is_valid_swap(S_set, remove_val, add_val):
    """Check if swapping remove_val with add_val in valid set S_set keeps it valid."""
    if add_val in S_set:
        return False
    if remove_val not in S_set:
        return False
    
    S_without = S_set - {remove_val}
    
    # Check: add_val does not divide sum of any 3 in S_without
    for b, c in combinations(S_without, 2):
        r = (-(b + c)) % add_val
        for a in _divs[add_val][r]:
            if a in S_without and a != b and a != c:
                return False
    
    # Check: for each a in S_without, a does not divide sum of add_val + 2 others
    for a in S_without:
        r = (-add_val) % a
        for b, c in _pairs_divs[a][r]:
            if b in S_without and c in S_without and a != b and a != c:
                return False
                
    return True


def _greedy(descending=True):
    """Greedy construction."""
    selected = []
    S_set = set()
    if descending:
        order = range(_N, 0, -1)
    else:
        order = range(1, _N + 1)
    for x in order:
        if time.time() > _DEADLINE:
            break
        if _is_valid_addition(S_set, x):
            selected.append(x)
            S_set.add(x)
    return selected


def _local_search(S, time_budget=30.0):
    """Local search with swaps and insertions to improve set size."""
    S = list(S)
    S_set = set(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE:
        improved = False
        
        # Try to add a random element not in S
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            for x in candidates[:100]: # Increased candidates due to speedup
                if time.time() > _DEADLINE:
                    return best
                if _is_valid_addition(S_set, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # Try swaps: remove one from S, add one not in S
        if len(S) >= 1:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            
            for a in to_remove[:60]: # Increased swaps
                if time.time() > _DEADLINE:
                    return best
                for x in candidates[:60]:
                    if time.time() > _DEADLINE:
                        return best
                    if _is_valid_swap(S_set, a, x):
                        # Perform swap
                        S.remove(a)
                        S.append(x)
                        S_set.remove(a)
                        S_set.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # If no improvement, perturb
        if not improved:
            if len(S) > 3:
                k = random.randint(1, 2)
                to_remove = random.sample(S, k)
                for v in to_remove:
                    S.remove(v)
                    S_set.remove(v)
    
    return best


def _multi_start_greedy(num_starts=30, time_budget=20.0):
    """Try multiple greedy orders with randomization."""
    best = []
    start_time = time.time()
    
    for _ in range(num_starts):
        if time.time() - start_time > time_budget or time.time() > _DEADLINE:
            break
        
        # Randomized greedy: shuffle order with bias toward larger
        order = list(range(1, _N + 1))
        weights = [random.random() * _N + v for v in order]
        order.sort(key=lambda v: -weights[v - 1])
        
        selected = []
        S_set = set()
        for x in order:
            if time.time() > _DEADLINE:
                break
            if _is_valid_addition(S_set, x):
                selected.append(x)
                S_set.add(x)
        
        if len(selected) > len(best):
            best = list(selected)
    
    return best


def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Basic greedy
    S1 = _greedy(descending=True)
    S2 = _greedy(descending=False)
    best = max([S1, S2], key=len)
    
    # Phase 2: Multi-start greedy
    if time.time() < _DEADLINE - 25:
        S3 = _multi_start_greedy(num_starts=50, time_budget=15.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 3: Local search from best
    if time.time() < _DEADLINE - 15:
        S4 = _local_search(best, time_budget=12.0)
        if len(S4) > len(best):
            best = S4
    
    # Phase 4: Another local search round if time permits
    if time.time() < _DEADLINE - 5:
        S5 = _local_search(best, time_budget=5.0)
        if len(S5) > len(best):
            best = S5
            
    return best
# EVOLVE-BLOCK-END
