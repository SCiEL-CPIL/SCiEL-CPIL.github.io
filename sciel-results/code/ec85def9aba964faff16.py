"""Model-and-optimize approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
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
    
    # Precompute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    # Initial solution: use the divisors themselves
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_assign = naive[:]
    
    def check_window(W, L, divisors, deadline):
        """Check if window [L, L+W] has a valid distinct assignment."""
        if L < 1 or L + W > MAX_V:
            return None
        
        # For each divisor, compute the list of offsets (relative to L) where multiples fall
        candidate_offsets = []
        for d in divisors:
            p = L % d
            first_offset = (d - p) % d
            if first_offset > W:
                return None  # No multiple in window
            offsets = list(range(first_offset, W + 1, d))
            candidate_offsets.append(offsets)
        
        # Quick check: each divisor must have at least one candidate
        # (already guaranteed by the above)
        
        # Use backtracking with MRV and forward checking
        assignment = [None] * n
        used_positions = set()
        
        # Order divisors by number of available positions (MRV)
        order = sorted(range(n), key=lambda i: len(candidate_offsets[i]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            # Available positions for divisor i (not used)
            avail = [j for j in candidate_offsets[i] if j not in used_positions]
            
            if not avail:
                return False
            
            # Sort by how many other unassigned divisors also want this position
            def conflict_count(pos):
                count = 0
                for k in range(idx + 1, n):
                    if pos in candidate_offsets[k]:
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
    for W in range(9, 100):
        if time.time() > deadline:
            break
        
        max_L = min(LCM, MAX_V - W)
        
        # Try all L values from 1 to max_L
        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            result = check_window(W, L, divisors, deadline)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found = True
                    break
        
        if found and best_val == W:
            break
    
    # Local search to polish
    if best_assign is not None and best_val > 9:
        no_improve = 0
        max_no_improve = 50
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            candidate = best_assign[:]
            improved = False
            
            idx = random.randint(0, n - 1)
            d = divisors[idx]
            current = candidate[idx]
            
            # Try larger range of moves
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = current + delta * d
                if new_val < 1 or new_val > MAX_V:
                    continue
                if any(candidate[j] == new_val for j in range(n) if j != idx):
                    continue
                
                candidate[idx] = new_val
                span = max(candidate) - min(candidate)
                if span < best_val:
                    best_val = span
                    best_assign = candidate[:]
                    improved = True
                    no_improve = 0
                    break
                candidate[idx] = current
            
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
    
    # Final validation
    if best_assign is None:
        best_assign = divisors[:]
    
    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)
    
    # Verify all constraints
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int) and not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n
    
    return result

# EVOLVE-BLOCK-END
