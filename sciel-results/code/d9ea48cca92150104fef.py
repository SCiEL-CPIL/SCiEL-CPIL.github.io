"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

def _is_valid_addition(S, x):
    """
    Check if adding x to valid set S keeps it valid.
    S is a list of distinct integers. x is not in S.
    Conditions:
    1. x does not divide sum of any 3 distinct elements in S.
    2. For each a in S, a does not divide (x + u + v) for any distinct u, v in S \ {a}.
    """
    n = len(S)
    if n < 3:
        for a in S:
            others = [v for v in S if v != a]
            for u, v in combinations(others, 2):
                if (x + u + v) % a == 0:
                    return False
        return True

    # Check 1: x does not divide sum of any 3 in S
    for t in combinations(S, 3):
        if sum(t) % x == 0:
            return False
    
    # Check 2: For each a in S, a does not divide x + u + v for u, v in S \ {a}
    for i, a in enumerate(S):
        target_res = (-x) % a
        others = [S[j] for j in range(n) if j != i]
        for u_idx in range(len(others)):
            u = others[u_idx]
            needed_res = (target_res - u) % a
            for v_idx in range(u_idx + 1, len(others)):
                v = others[v_idx]
                if v % a == needed_res:
                    return False
    return True

def _valid_set(S):
    """Full validation, used for final check or initialization."""
    S = list(S)
    n = len(S)
    for i, a in enumerate(S):
        others = [S[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_large_first():
    """
    Greedy construction starting from largest numbers.
    Large numbers are less likely to divide sums of others because the sums are bounded.
    """
    selected = []
    for x in range(_N, 0, -1):
        if _is_valid_addition(selected, x):
            selected.append(x)
        if time.time() > _DEADLINE - 5:
            break
    return selected

def _local_search(S, time_budget=30.0):
    """
    Local search with 1-swap, 2-swap, and insertion.
    """
    S = list(S)
    start_time = time.time()
    best = list(S)
    best_len = len(S)
    
    all_vals = set(range(1, _N + 1))
    
    while time.time() - start_time < time_budget and time.time() < _DEADLINE - 5:
        improved = False
        
        # 1. Try Insertion
        if len(S) < _MAX:
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            for x in candidates[:20]:
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best = list(S)
                    break
        
        # 2. Try 1-Swap
        if not improved and len(S) > 0:
            to_remove = list(S)
            random.shuffle(to_remove)
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            
            for a in to_remove[:10]:
                for x in candidates[:10]:
                    S_minus_a = [v for v in S if v != a]
                    if _is_valid_addition(S_minus_a, x):
                        S = S_minus_a + [x]
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best = list(S)
                        break
                if improved:
                    break
        
        # 3. Try 2-Swap
        if not improved and len(S) >= 2:
            to_remove = random.sample(S, 2)
            candidates = list(all_vals - set(S))
            random.shuffle(candidates)
            
            for x1 in candidates[:5]:
                for x2 in candidates[:5]:
                    if x1 == x2: continue
                    S_minus_2 = [v for v in S if v not in to_remove]
                    if _is_valid_addition(S_minus_2, x1):
                        S_temp = S_minus_2 + [x1]
                        if _is_valid_addition(S_temp, x2):
                            S = S_temp + [x2]
                            improved = True
                            if len(S) > best_len:
                                best_len = len(S)
                                best = list(S)
                            break
                if improved:
                    break
        
        # 4. Perturbation if stuck
        if not improved:
            if len(S) > 5:
                to_remove = random.sample(S, 3)
                S = [v for v in S if v not in to_remove]
                candidates = [v for v in range(_N, 1, -1) if v not in S]
                random.shuffle(candidates)
                for x in candidates[:3]:
                    if _is_valid_addition(S, x):
                        S.append(x)
            else:
                S = _greedy_large_first()
                if len(S) > best_len:
                    best_len = len(S)
                    best = list(S)
    
    return best

def _multi_start_greedy(num_starts=20, time_budget=20.0):
    """Try multiple greedy orders with randomization."""
    best = []
    start_time = time.time()
    
    for _ in range(num_starts):
        if time.time() - start_time > time_budget or time.time() > _DEADLINE - 10:
            break
        
        order = list(range(1, _N + 1))
        weights = [random.random() * _N + v for v in order]
        order.sort(key=lambda v: -weights[v-1])
        
        selected = []
        for x in order:
            if _is_valid_addition(selected, x):
                selected.append(x)
        
        if len(selected) > len(best):
            best = list(selected)
    
    return best

def construct():
    """Main entry point: find maximum valid set."""
    
    # Phase 1: Greedy from large to small
    best = _greedy_large_first()
    
    # Phase 2: Multi-start greedy
    if time.time() < _DEADLINE - 30:
        S3 = _multi_start_greedy(num_starts=30, time_budget=15.0)
        if len(S3) > len(best):
            best = S3
    
    # Phase 3: Local search
    if time.time() < _DEADLINE - 10:
        S_ls = _local_search(best, time_budget=30.0)
        if len(S_ls) > len(best):
            best = S_ls
            
    # Phase 4: Another local search pass with different random seed
    if time.time() < _DEADLINE - 5:
        random.seed(int(time.time() * 1000) % 10000)
        S_ls2 = _local_search(best, time_budget=5.0)
        if len(S_ls2) > len(best):
            best = S_ls2
    
    return best
# EVOLVE-BLOCK-END
