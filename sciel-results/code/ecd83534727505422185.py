import math
import time

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    
    # Precompute for each divisor, the residue classes mod L
    # Actually, multiples of d in [0, L-1] are: 0, d, 2d, ..., (L/d - 1)*d
    # Number of multiples of d in one period: L // d
    
    def find_sdr_fast(candidates, n_vars):
        """
        candidates: list of lists of possible values for each variable.
        Returns a list of distinct values or None.
        Uses bitmask for used set.
        """
        assignment = [0] * n_vars
        used_mask = 0
        # We'll track which global index each value maps to
        # But values can be large, so we need a different approach
        # Actually, let's just use a set but with a faster backtracking order
        
        # Sort by constraint level
        order = sorted(range(n_vars), key=lambda i: len(candidates[i]))
        
        # Pre-sort candidates within each variable
        for i in range(n_vars):
            candidates[i].sort()
        
        def backtrack(idx):
            if idx == n_vars:
                return True
            
            var = order[idx]
            used = used_mask  # can't easily use bitmask for arbitrary values
            # Use a local set for this level
            for val in candidates[var]:
                if val not in used_set:
                    assignment[var] = val
                    used_set.add(val)
                    if backtrack(idx + 1):
                        return True
                    used_set.discard(val)
                    assignment[var] = 0
            return False
        
        used_set = set()
        if backtrack(0):
            return assignment
        return None

    def is_feasible(R):
        # For each start s in [0, L-1], check if we can pick distinct multiples
        # in [s, s+R] for each divisor.
        # Optimization: sort s values by how "central" they are or by some heuristic
        # Actually, just iterate all s
        
        for s in range(L):
            s_end = s + R
            candidates = []
            feasible_s = True
            for i, d in enumerate(divisors):
                k_min = (s + d - 1) // d
                k_max = s_end // d
                
                if k_min > k_max:
                    feasible_s = False
                    break
                
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(vals)
            
            if not feasible_s:
                continue
            
            # Check if we can assign distinct values
            # Use backtracking with MRV (minimum remaining values)
            n_vars = len(candidates)
            if any(len(c) == 0 for c in candidates):
                continue
            
            # Quick check: if total candidates < n_vars, impossible
            # (but this is weak)
            
            assignment = [0] * n_vars
            used_set = set()
            
            # Order variables by fewest candidates
            var_order = sorted(range(n_vars), key=lambda i: len(candidates[i]))
            
            # Additional pruning: if any variable has only 1 candidate,
            # that value is forced
            
            def backtrack(idx):
                if idx == n_vars:
                    return True
                
                var = var_order[idx]
                cands = candidates[var]
                
                for val in cands:
                    if val not in used_set:
                        assignment[var] = val
                        used_set.add(val)
                        if backtrack(idx + 1):
                            return True
                        used_set.remove(val)
                        assignment[var] = 0
                return False
            
            if backtrack(0):
                sol = assignment
                # Verify
                if len(set(sol)) != n:
                    continue
                for i, x in enumerate(sol):
                    if x <= 0 or x > max_val:
                        break
                    if x % divisors[i] != 0:
                        break
                else:
                    return sol
        
        return None

    # Binary search for minimum R
    low = 0
    high = L  # safe upper bound
    best_sol = None
    best_R = L + 1
    
    start_time = time.time()
    deadline = start_time + 50  # leave margin
    
    while low <= high and time.time() < deadline:
        mid = (low + high) // 2
        sol = is_feasible(mid)
        if sol is not None:
            best_sol = sol
            best_R = mid
            high = mid - 1
        else:
            low = mid + 1
    
    if best_sol is None:
        # Fallback: use the divisors themselves
        best_sol = divisors[:]
    
    return [int(x) for x in best_sol]
# EVOLVE-BLOCK-END
