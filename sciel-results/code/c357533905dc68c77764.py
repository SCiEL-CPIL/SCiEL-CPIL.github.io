"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_sidon_cover(S):
    """Check if S is a valid Sidon set containing 0 with all differences 1..30."""
    S = list(S)
    if 0 not in S:
        return False
    if len(S) > 32:
        return False
    if any(x < 0 or x > 10**6 for x in S):
        return False
    if len(set(S)) != len(S):
        return False
    # Compute all positive differences
    diffs = set()
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            d = S[j] - S[i]
            if d in diffs:
                return False
            diffs.add(d)
    # Check all differences 1..30 are present
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _cost(S):
    """Cost function: maximize coverage of 1..30, penalize duplicates and large max."""
    S = list(S)
    if len(set(S)) != len(S):
        return float('inf')
    diffs = []
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            d = S[j] - S[i]
            diffs.append(d)
    diff_set = set(diffs)
    # Penalty for duplicate differences
    dup_penalty = len(diffs) - len(diff_set)
    # Coverage of 1..30
    coverage = sum(1 for d in range(1, 31) if d in diff_set)
    # Penalize large max element
    max_elem = max(S) if S else 0
    return (30 - coverage) * 1000 + dup_penalty * 100 + max_elem

def _greedy_build():
    """Greedy construction as a fallback."""
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return [s[j] - s[i] for i in range(len(s)) for j in range(i+1, len(s))]
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        # Try adding one element
        for y in sorted(set(v + dd for v in a for dd in (d, -d) if v + dd >= 0) - set(a)):
            trial = sorted(a + [y])
            diffs = get_diffs(trial)
            if len(set(diffs)) == len(diffs):
                a = trial
                found = True
                break
        if not found:
            # Try adding two elements
            base = 2 * max(a) + 3 * d + 1
            y = base
            while True:
                trial = sorted(a + [y, y + d])
                diffs = get_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    break
                y += 1
    return a

def _simulated_annealing(seed=42, time_budget=10.0):
    """Simulated annealing to find a good Sidon set."""
    random.seed(seed)
    start = time.time()
    deadline = start + time_budget
    
    # Start with greedy solution
    S = _greedy_build()
    current_cost = _cost(S)
    best_S = list(S)
    best_cost = current_cost
    
    n = len(S)
    temp = 1000.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate neighbor: move one element
        idx = random.randint(0, n - 1)
        if idx == 0 and S[0] == 0:
            # Don't move 0
            idx = random.randint(1, n - 1)
        old_val = S[idx]
        # Try small moves
        step = random.choice([-3, -2, -1, 1, 2, 3, -10, 10, -20, 20, -50, 50])
        new_val = old_val + step
        if new_val < 0:
            new_val = old_val - step
        if new_val > 500:
            continue
        
        S[idx] = new_val
        new_S = sorted(S)
        # Check for duplicates
        if len(set(new_S)) != len(new_S):
            S[idx] = old_val
            continue
        
        new_cost = _cost(new_S)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_S = list(new_S)
        else:
            # Accept with probability exp(-(delta)/temp)
            delta = new_cost - current_cost
            if delta < 0:
                S[idx] = new_val
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_S = list(new_S)
            else:
                import math
                if delta < 10000:  # Avoid overflow
                    prob = math.exp(-delta / max(temp, 1e-10))
                    if random.random() < prob:
                        S[idx] = new_val
                        current_cost = new_cost
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    return best_S

def _local_optimize(S, time_budget=5.0):
    """Local optimization: try to reduce max element by moving elements."""
    start = time.time()
    deadline = start + time_budget
    S = sorted(list(S))
    best_S = list(S)
    best_max = max(S)
    
    while time.time() < deadline:
        improved = False
        for i in range(len(S)):
            if S[i] == 0:
                continue
            # Try moving this element to a smaller value
            old_val = S[i]
            for new_val in range(1, old_val):
                if new_val in S:
                    continue
                trial = S.copy()
                trial[i] = new_val
                trial = sorted(trial)
                if len(set(trial)) != len(trial):
                    continue
                if _is_valid_sidon_cover(trial):
                    if max(trial) < best_max:
                        best_max = max(trial)
                        best_S = trial
                        S = trial
                        improved = True
                        break
        if not improved:
            break
    
    return best_S

def _backtrack_small():
    """Backtracking search for small solutions."""
    start_time = time.time()
    deadline = start_time + 30.0
    
    best_result = None
    best_max_val = float('inf')
    
    def dfs(S, used_diffs, required_remaining, max_val_so_far):
        nonlocal best_result, best_max_val
        
        if time.time() > deadline:
            return
        
        if not required_remaining:
            if max_val_so_far < best_max_val:
                best_max_val = max_val_so_far
                best_result = list(S)
            return
        
        if max_val_so_far >= best_max_val:
            return
        
        target_diff = min(required_remaining)
        
        candidates = set()
        for y in S:
            candidates.add(y + target_diff)
            if y - target_diff >= 0:
                candidates.add(y - target_diff)
        
        candidates = sorted(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                return
            if x in S:
                continue
            if x >= best_max_val:
                continue
            
            new_S = sorted(S | {x})
            new_diffs = set()
            valid = True
            for i in range(len(new_S)):
                for j in range(i+1, len(new_S)):
                    d = new_S[j] - new_S[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_max = max(new_S)
            if new_max >= best_max_val:
                continue
            
            new_required = required_remaining - new_diffs
            dfs(frozenset(new_S), used_diffs | new_diffs, new_required, new_max)
    
    initial_S = frozenset([0])
    initial_used_diffs = set()
    initial_required = set(range(1, 31))
    
    dfs(initial_S, initial_used_diffs, initial_required, 0)
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Strategy: Multiple approaches, return the best valid solution found.
    candidates = []
    
    # 1. Try backtracking (fast for small problems)
    if time.time() < deadline:
        bt_result = _backtrack_small()
        if bt_result is not None:
            if _is_valid_sidon_cover(bt_result):
                candidates.append(bt_result)
    
    # 2. Try simulated annealing
    if time.time() < deadline - 15.0:
        sa_result = _simulated_annealing(seed=42, time_budget=8.0)
        if _is_valid_sidon_cover(sa_result):
            candidates.append(sa_result)
        else:
            # Try to fix it with local optimization
            fixed = _local_optimize(sa_result, time_budget=5.0)
            if _is_valid_sidon_cover(fixed):
                candidates.append(fixed)
    
    # 3. Try different seeds for SA
    if time.time() < deadline - 10.0:
        for seed in [1, 7, 123, 999]:
            if time.time() > deadline - 5.0:
                break
            sa_result = _simulated_annealing(seed=seed, time_budget=3.0)
            if _is_valid_sidon_cover(sa_result):
                candidates.append(sa_result)
    
    # 4. Greedy fallback
    greedy = _greedy_build()
    if _is_valid_sidon_cover(greedy):
        candidates.append(greedy)
    
    # Pick the best candidate (smallest max element)
    if candidates:
        best = min(candidates, key=lambda s: max(s))
        return best
    else:
        # Last resort: return greedy even if not perfectly valid (shouldn't happen)
        return greedy

# EVOLVE-BLOCK-END
