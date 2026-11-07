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
    
    # We want to find the minimum range R such that there exist distinct multiples
    # m_i = k_i * d_i in [s, s+R] for some s.
    
    # The problem is equivalent to finding a set of residues r_i mod d_i (which must be 0)
    # such that the values can be packed into an interval of length R.
    
    # Since r_i = 0 for all i, the values are exactly the multiples.
    
    # We can search for R by checking feasibility.
    # A faster way is to use the structure of the multiples.
    
    # Let's precompute the "gaps" or structure.
    # Actually, the binary search + backtracking approach is robust.
    # The bottleneck is the backtracking for large R (many candidates).
    # But we start with small R in binary search? No, we start mid.
    # If we start with small R, it's infeasible quickly.
    
    # Optimization: Use a tighter upper bound.
    # The divisors are [6, 10, 14, 15, 21, 22, 26, 33, 35, 39].
    # The trivial solution is the divisors themselves: range 33.
    # Can we do better?
    # Let's try to find a solution with R < 33.
    
    # Let's just run the binary search from 1 to 33 (or slightly higher to be safe, e.g., 50).
    # If we find a solution with R < 33, great. If not, we return the divisors.
    
    def find_sdr(candidates):
        n_vars = len(candidates)
        assignment = [None] * n_vars
        used = set()
        
        order = sorted(range(n_vars), key=lambda i: len(candidates[i]))
        
        def backtrack(idx):
            if idx == n_vars:
                return True
            
            var = order[idx]
            for val in candidates[var]:
                if val not in used:
                    assignment[var] = val
                    used.add(val)
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[var] = None
            return False
            
        if backtrack(0):
            return assignment
        return None

    def is_feasible(R):
        for s in range(L):
            candidates = []
            feasible_s = True
            for i, d in enumerate(divisors):
                k_min = (s + d - 1) // d
                k_max = (s + R) // d
                
                if k_min > k_max:
                    feasible_s = False
                    break
                
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(vals)
            
            if not feasible_s:
                continue
            
            sol = find_sdr(candidates)
            if sol is not None:
                max_sol = max(sol)
                if max_sol > max_val:
                    shift = (max_sol - max_val + L - 1) // L
                    sol = [x - shift * L for x in sol]
                    if min(sol) <= 0:
                        continue
                
                if len(set(sol)) != n:
                    continue
                for i, x in enumerate(sol):
                    if x <= 0 or x > max_val:
                        continue
                    if x % divisors[i] != 0:
                        continue
                
                return sol
        
        return None

    # We know R=33 is feasible (the divisors themselves).
    # Let's search for R in [1, 32].
    # If none found, return divisors.
    
    best_sol = None
    best_R = 33
    fallback = divisors[:]
    
    # Binary search for minimum R in [1, 32]
    low = 1
    high = 32
    start_time = time.time()
    
    while low <= high:
        if time.time() - start_time > 50:
            break
            
        mid = (low + high) // 2
        sol = is_feasible(mid)
        if sol is not None:
            best_sol = sol
            best_R = mid
            high = mid - 1
        else:
            low = mid + 1
            
    if best_sol is None:
        best_sol = fallback
    else:
        # Verify best_sol
        if len(set(best_sol)) != n:
            best_sol = fallback
        else:
            for i, x in enumerate(best_sol):
                if x <= 0 or x > max_val or x % divisors[i] != 0:
                    best_sol = fallback
                    break
    
    return [int(x) for x in best_sol]
# EVOLVE-BLOCK-END
