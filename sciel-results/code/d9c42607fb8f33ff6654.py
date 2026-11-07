import time
import math
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill
    
    # LCM of all divisors
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm(lcm_val, d)
    # lcm_val = 30030
    
    best_range = float('inf')
    best_vals = None
    
    # For each divisor d, the multiples in [A, A+W] are:
    # floor(A/d)*d, (floor(A/d)+1)*d, ... up to <= A+W
    # We need to pick exactly one multiple per divisor, all distinct, all in [A, A+W]
    
    # Strategy: iterate over window size W from small to large.
    # For each W, iterate over A in [0, lcm_val-1] (periodicity).
    # For each (A, W), check if we can assign distinct multiples.
    
    # To speed up: for a given (A, W), each divisor d has at most 
    # floor(W/d) + 1 candidate multiples in the window.
    # Use backtracking with constraint propagation.
    
    # Precompute: for each A mod lcm_val, and each W, the candidates.
    # But that's too much. Instead, iterate W and A.
    
    # Optimize: for a given W, the number of candidates per divisor is small.
    # For W < min(divisors) = 6, each divisor has at most 1 candidate (if any).
    # Actually for divisor d, candidates in [A, A+W] are multiples of d in that range.
    
    # Let's think about it differently:
    # For a fixed window [A, A+W], divisor d can use any multiple of d in [A, A+W].
    # The number of such multiples is at most W//d + 1.
    
    # For small W, this is very constrained. Let's search W from 1 upward.
    
    # For each W, for each A in range(lcm_val), try to find an assignment.
    # But lcm_val = 30030, and W can be up to... let's see.
    # The current best is 1.7368, which means range = seed/score. 
    # If seed = 50 (typical), range ≈ 28.8. Let's assume seed is around 50.
    # Actually, score = seed/value, so value = seed/score.
    # If current score is 1.7368 and we don't know seed, let's just search.
    
    # Better approach: for each W, we can be smarter.
    # For a given W, a divisor d has candidates only if there's at least one multiple of d in [A, A+W].
    # The number of A values where divisor d has NO multiple in [A, A+W] is... 
    # Actually, for any A, there's always at least one multiple of d in [A, A+d-1], so for W >= d-1, every A works.
    # For W < d-1, some A values won't have a multiple of d.
    
    # Let me just do a direct search with backtracking for each (A, W).
    
    # To make it efficient, iterate W from 1 to some max, and for each W,
    # iterate A over a reduced set.
    
    # Key insight: for a fixed W, the pattern of which divisors have candidates
    # repeats with period lcm_val. But we can also note that for a fixed W,
    # we only need to check A in [0, lcm_val - 1].
    
    # However, 30030 * (max_W) iterations with backtracking might be slow.
    # Let's add pruning: if W is too small, skip.
    
    # Minimum possible W: we need 10 distinct values in [A, A+W], so W >= 9.
    # Also, each divisor must have at least one candidate.
    
    # Let's iterate W from 9 upward, and for each W, iterate A from 0 to lcm_val-1.
    # For each (A, W), do backtracking.
    
    # To speed up backtracking: order divisors by number of candidates (ascending).
    
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        for A in range(lcm_val):
            if time.time() > deadline:
                break
            
            # Get candidates for each divisor
            candidates = []
            for d in divisors:
                # Multiples of d in [A, A+W]
                first = ((A + d - 1) // d) * d
                if first > A + W:
                    candidates.append([])
                    break
                cands = []
                m = first
                while m <= A + W:
                    cands.append(m)
                    m += d
                candidates.append(cands)
            else:
                # All divisors have at least one candidate
                # Check if total distinct values possible
                # Backtracking
                # Order by fewest candidates first
                order = sorted(range(n), key=lambda i: len(candidates[i]))
                
                # If any has 0 candidates, skip (already handled)
                if any(len(candidates[i]) == 0 for i in order):
                    continue
                
                # If the sum of min candidates... just do backtracking
                used = set()
                assignment = [None] * n
                
                def backtrack(idx):
                    nonlocal best_range, best_vals
                    if idx == n:
                        # Found a valid assignment
                        vals = assignment[:]
                        r = max(vals) - min(vals)
                        if r < best_range:
                            best_range = r
                            best_vals = vals[:]
                        return True  # found one, could stop or continue
                    i = order[idx]
                    for v in candidates[i]:
                        if v not in used:
                            used.add(v)
                            assignment[i] = v
                            if backtrack(idx + 1):
                                # We found a solution, but might want to find better
                                # For now, return to try other W values
                                return False  # continue searching
                            used.discard(v)
                            assignment[i] = None
                    return False
                
                backtrack(0)
                
                # If we found a solution with range W, we can break inner loop
                # since we're searching in order of W
                if best_range <= W:
                    # Can't do better than W for this W value
                    # But we might find a smaller range with different A? No, range <= W by construction
                    # Actually range could be less than W. But we're iterating W upward,
                    # so if best_range <= W, we've found the best for this W.
                    # We should still check other A for this W to see if we can get range < W
                    # But since we're iterating W upward, and we want minimum range,
                    # let's just continue to next W if best_range is small enough.
                    pass
        
        # If best_range is small, we can stop
        if best_range is not None and best_range <= W:
            # We found a solution with range <= W, and we're searching W upward
            # The optimal is at most W. We should verify no smaller W works.
            # But we already checked all smaller W. So this is optimal.
            break
    
    if best_vals is None:
        # Fallback: use the divisors themselves
        return divisors[:]
    
    # Ensure all are Python ints
    return [int(v) for v in best_vals]
