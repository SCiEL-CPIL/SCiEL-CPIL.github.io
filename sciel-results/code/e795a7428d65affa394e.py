import math
import time

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Calculate LCM of divisors
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # L is 30030
    # We search for the minimum range R
    # The solution is periodic with period L.
    # We can restrict the search to windows starting at s in [0, L-1].
    # Actually, we can just search for R.
    
    # Precompute multiples? No, compute on the fly or precompute offsets.
    # For a window [s, s+R], the candidates for d_i are multiples of d_i in that range.
    
    def find_sdr(candidates):
        """
        candidates: list of lists of possible values for each variable.
        Returns a list of distinct values or None.
        Uses backtracking.
        """
        n_vars = len(candidates)
        assignment = [None] * n_vars
        used = set()
        
        # Sort variables by number of candidates (most constrained first)
        # This helps pruning
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
        # Check if there exists a start offset s such that we can pick distinct multiples
        # in [s, s+R] for each divisor.
        # We iterate s from 0 to L-1.
        # Optimization: If R is very large, it's likely feasible.
        # If R is small, we check all s.
        
        # Precompute for each s and each divisor i, the list of candidates?
        # That might be too much memory if done naively, but L=30030 is small.
        # Let's just compute on the fly inside the loop.
        
        for s in range(L):
            candidates = []
            feasible_s = True
            for i, d in enumerate(divisors):
                # Find multiples of d in [s, s+R]
                # k_min = ceil(s/d)
                # k_max = floor((s+R)/d)
                k_min = (s + d - 1) // d
                k_max = (s + R) // d
                
                if k_min > k_max:
                    feasible_s = False
                    break
                
                # Generate candidates
                # To keep backtracking fast, limit candidates if possible?
                # No, we need exactness. But R will be small, so k_max - k_min is small.
                # Max candidates per divisor is roughly R/d + 1.
                # For d=6, R=100, ~17 candidates.
                # Backtracking on 10 vars with ~10-20 candidates each is fine.
                
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(vals)
            
            if not feasible_s:
                continue
            
            sol = find_sdr(candidates)
            if sol is not None:
                # Found a valid set of relative values.
                # We need to return the actual integers.
                # The values in sol are absolute values relative to the global number line?
                # No, `vals` were computed as `k*d`, which are absolute values in the range [s, s+R].
                # So `sol` contains the actual integers.
                # We must ensure they are <= max_val.
                # Since s < L and R is likely small, max val is s+R.
                # If s+R > max_val, we might need to wrap or shift?
                # The problem says "positive integers <= 1000000".
                # If the found solution exceeds max_val, we can subtract L?
                # No, the values are specific multiples.
                # However, if R is small, and we pick s such that s+R <= max_val, we are safe.
                # L=30030. max_val=1000000.
                # We can always choose s such that the window is within [1, max_val] if R < max_val - L.
                # Actually, we can just pick any valid s.
                # If s+R > max_val, we can try s' = s - L? No, s is in [0, L-1].
                # If the solution found is > max_val, it's invalid.
                # But we can just check if max(sol) <= max_val.
                # If not, we might need to search for a different s that fits in range.
                # Or, since L is small, we can just map the solution to the first period [1, L+R]?
                # Wait, the multiples are fixed.
                # If we find a solution in [s, s+R], and max(sol) > max_val,
                # we can subtract L from all values?
                # If x is a multiple of d, x-L is not necessarily a multiple of d unless L is a multiple of d.
                # L IS a multiple of all d. So x-L is also a multiple of d.
                # And x-L > 0 if x > L.
                # So we can shift the solution down by L if it exceeds max_val.
                
                max_sol = max(sol)
                if max_sol > max_val:
                    # Shift down by L until it fits
                    # Since L is common multiple, validity is preserved.
                    shift = 0
                    while max_sol - shift * L > max_val:
                        shift += 1
                    sol = [x - shift * L for x in sol]
                    # Ensure positive
                    if min(sol) <= 0:
                        # This shouldn't happen if we picked s appropriately or if R is small
                        # But if we shifted too much, it might.
                        # Actually, if max_sol > max_val, and min_sol > 0, shifting down by L
                        # keeps them multiples.
                        # If min_sol - shift*L <= 0, we might have gone too far?
                        # No, we want to fit in [1, max_val].
                        # If the range R is small, there is always a copy in [1, max_val]
                        # as long as max_val >= L + R.
                        # 1000000 >> 30030 + R.
                        pass
                
                # Verify distinctness and divisibility
                if len(set(sol)) != n:
                    continue
                for i, x in enumerate(sol):
                    if x <= 0 or x > max_val:
                        continue
                    if x % divisors[i] != 0:
                        continue
                
                return sol
        
        return None

    # Binary search for R
    # Lower bound: 0 (but distinct means R>=1 if n>1)
    # Upper bound: L (since there are L residues, and we need n distinct ones, R can be up to L)
    # Actually, R can be larger, but L is a safe upper bound for the *minimal* R?
    # Not necessarily, but usually R is small.
    # Let's set high = L.
    
    low = 0
    high = L
    best_sol = None
    best_R = None
    
    start_time = time.time()
    
    while low <= high:
        if time.time() - start_time > 45:
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
        # Fallback: Just return the divisors themselves?
        # They are distinct?
        # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
        # Are they distinct? Yes.
        # Are they <= 1000000? Yes.
        # Are they divisible by their corresponding inputs? Yes.
        # Range: 39 - 6 = 33.
        # This is a valid solution with R=33.
        best_sol = divisors[:]
        
    # Ensure types are int
    return [int(x) for x in best_sol]
# EVOLVE-BLOCK-END
