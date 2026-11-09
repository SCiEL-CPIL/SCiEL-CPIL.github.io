import sys
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    LCM = 30030
    
    # Helper to get candidates in [L, L+W] for a divisor d
    def get_candidates(L, W, d):
        # First multiple >= L
        start_k = (L + d - 1) // d
        # Last multiple <= L+W
        end_k = (L + W) // d
        if start_k > end_k:
            return []
        return [k * d for k in range(start_k, end_k + 1)]

    # Backtracking solver
    def solve_matching(cands):
        # cands: list of lists of candidates for each divisor
        # Sort indices by number of candidates (MRV heuristic)
        indices = sorted(range(len(cands)), key=lambda i: len(cands[i]))
        
        assigned = {}
        used_values = set()
        
        def backtrack(idx):
            if idx == len(indices):
                return True
            
            div_idx = indices[idx]
            candidates = cands[div_idx]
            
            for val in candidates:
                if val not in used_values:
                    used_values.add(val)
                    assigned[div_idx] = val
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(val)
                    del assigned[div_idx]
            return False
        
        if backtrack(0):
            # Reconstruct result in original order
            result = [0] * len(cands)
            for i in range(len(cands)):
                result[i] = assigned[i]
            return result
        return None

    # Search for minimum W
    # We know W must be at least 9 (10 distinct ints)
    # Upper bound: random guessing suggests < 500, but let's go up to 1000 to be safe
    # Time limit is 60s. 30030 windows * 1000 W * fast solver is fine.
    
    deadline = time.time() + 50 # 10s safety margin
    
    for W in range(10, 1000):
        for L in range(LCM):
            if time.time() > deadline:
                # Fallback: return a valid (though not optimal) solution
                # Construct a valid solution with large W
                # Just use the divisors themselves scaled to be distinct and large?
                # No, they must be multiples.
                # Simple valid solution: x_i = d_i * 1000. Distinct? 
                # 6000, 10000, 14000... Yes. Max=39000, Min=6000. W=33000.
                return [d * 1000 for d in divisors]
            
            cands = [get_candidates(L, W, d) for d in divisors]
            
            # Quick check: if any list is empty, skip
            if any(len(c) == 0 for c in cands):
                continue
                
            # Optional: Hall's theorem check for small sets? 
            # The backtracker is fast enough.
            
            sol = solve_matching(cands)
            if sol is not None:
                return sol
                
    # If loop finishes without finding (shouldn't happen for W < 1000)
    return [d * 1000 for d in divisors]
