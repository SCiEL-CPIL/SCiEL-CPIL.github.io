import time

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _find_solution_for_span(S):
    """
    Scans all L from 1 to MAX_VAL to find a window [L, L+S] 
    containing distinct multiples for all divisors.
    Returns list of 10 ints if found, else None.
    """
    d = DIVISORS
    # Pre-calculate divisors for speed
    for L in range(1, MAX_VAL + 1):
        if L + S > MAX_VAL:
            break
        
        # Check if each divisor has at least one multiple in [L, L+S]
        candidates = []
        feasible = True
        for i in range(N):
            di = d[i]
            start_k = (L + di - 1) // di
            end_k = (L + S) // di
            
            if start_k > end_k:
                feasible = False
                break
            
            # Generate candidate values
            # Since S is small, the number of candidates is small
            vals = []
            # Optimization: if range is large, we still only need a few
            # But S is small in our search loop
            for k in range(start_k, end_k + 1):
                vals.append(k * di)
            candidates.append(vals)
            
        if not feasible:
            continue
            
        # Now we need to pick one value from each candidates[i] 
        # such that all picked values are distinct.
        # This is a bipartite matching / backtracking problem.
        # Given N=10 and small domains, simple backtracking is fine.
        
        # Sort indices by domain size to prune early
        indices = list(range(N))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [None] * N
        
        def backtrack(idx_idx):
            if idx_idx == N:
                return True
            i = indices[idx_idx]
            for v in candidates[i]:
                if v not in used:
                    used.add(v)
                    assignment[i] = v
                    if backtrack(idx_idx + 1):
                        return True
                    used.remove(v)
            return False
            
        if backtrack(0):
            return assignment
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Seed solution: the divisors themselves
    seed_sol = DIVISORS[:]
    seed_span = max(seed_sol) - min(seed_sol)
    
    best_sol = seed_sol
    best_span = seed_span
    
    # Try to find smaller spans
    # We know seed span is 33. We try S from 1 to 32.
    # If we find a solution for S, it's better than 33.
    # We must check all L for a given S to be sure.
    
    # Optimization: If we are running out of time, return best found.
    for S in range(1, best_span):
        if time.time() > deadline - 5.0:
            break
            
        # Scan L
        # To speed up, we can check time periodically
        found = _find_solution_for_span(S)
        if found:
            best_sol = found
            best_span = S
            # Since we iterate S upwards, the first S found is the minimum
            # So we can break immediately?
            # Yes, because we want the MINIMUM span.
            # If we found a solution for S=5, we don't care about S=6.
            # And we checked S=1..4 and found nothing.
            # So S=5 is the global minimum (among those < seed).
            break
            
    # Ensure types are correct
    return [int(x) for x in best_sol]
# EVOLVE-BLOCK-END
