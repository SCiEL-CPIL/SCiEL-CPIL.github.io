"""Model-and-optimize approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, exp
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_assign = naive[:]
    
    def check_window(W, L):
        """Check if window [L, L+W] has a valid distinct assignment."""
        if L < 1 or L + W > MAX_V:
            return None
        
        candidate_offsets = []
        for d in divisors:
            p = L % d
            first_offset = (d - p) % d
            if first_offset > W:
                return None
            offsets = list(range(first_offset, W + 1, d))
            candidate_offsets.append(offsets)
        
        candidate_offsets_set = [set(offsets) for offsets in candidate_offsets]
        
        assignment = [None] * n
        used_positions = set()
        
        order = sorted(range(n), key=lambda i: len(candidate_offsets[i]))
        
        def backtrack(idx):
            if idx == n:
                return True
            
            i = order[idx]
            avail = [j for j in candidate_offsets[i] if j not in used_positions]
            
            if not avail:
                return False
            
            def conflict_count(pos):
                count = 0
                for k in range(idx + 1, n):
                    if pos in candidate_offsets_set[k]:
                        count += 1
                return count
            
            avail.sort(key=conflict_count)
            
            for j in avail:
                used_positions.add(j)
                assignment[i] = L + j
                if backtrack(idx + 1):
                    return True
                used_positions.discard(j)
                assignment[i] = None
            
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Systematic search over W from theoretical minimum
    for W in range(9, 30):
        if time.time() > deadline:
            break
        
        # For each feasible L, check if we can assign
        # Feasible L: for all d, (L % d) gives first_offset <= W
        max_L = min(LCM, MAX_V - W)
        
        found = False
        # Sample L values to speed up search
        stride = max(1, max_L // 5000)  # Check at most ~5000 L values
        for L in range(1, max_L + 1, stride):
            if time.time() > deadline:
                break
            
            # Quick check: all divisors must have a multiple in [L, L+W]
            valid = True
            for d in divisors:
                p = L % d
                first_offset = (d - p) % d
                if first_offset > W:
                    valid = False
                    break
            if not valid:
                continue
            
            result = check_window(W, L)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found = True
                    break
        
        if found and best_val == W:
            break
    
    # Local search with multiple restarts and larger steps
    if best_assign is not None:
        current_sol = best_assign[:]
        current_cost = max(current_sol) - min(current_sol)
        best_sol = current_sol[:]
        best_cost = current_cost
        
        for sa_run in range(3):
            if time.time() > deadline:
                break
            
            if sa_run == 0:
                current_sol = best_sol[:]
                current_cost = best_cost
            else:
                current_sol = best_sol[:]
                for _ in range(10):
                    idx = random.randint(0, n - 1)
                    d = divisors[idx]
                    delta = random.randint(-50, 50)
                    new_val = current_sol[idx] + delta * d
                    if 1 <= new_val <= MAX_V and all(current_sol[j] != new_val for j in range(n) if j != idx):
                        current_sol[idx] = new_val
                current_cost = max(current_sol) - min(current_sol)
            
            start_temp = 100.0
            end_temp = 0.01
            cooling_rate = 0.995
            temp = start_temp
            
            while temp > end_temp and time.time() < deadline:
                idx = random.randint(0, n - 1)
                d = divisors[idx]
                old_val = current_sol[idx]
                
                delta = random.randint(-50, 50)
                new_val = old_val + delta * d
                
                if new_val < 1 or new_val > MAX_V:
                    continue
                    
                if any(current_sol[j] == new_val for j in range(n) if j != idx):
                    continue
                
                new_sol = current_sol[:]
                new_sol[idx] = new_val
                new_cost = max(new_sol) - min(new_sol)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < exp(-delta_cost / temp):
                    current_sol = new_sol
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_sol = current_sol[:]
                
                temp *= cooling_rate
        
        if best_cost < best_val:
            best_val = best_cost
            best_assign = best_sol
    
    if best_assign is None:
        best_assign = divisors[:]
    
    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)
    
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int) and not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n
    
    return result

# EVOLVE-BLOCK-END
