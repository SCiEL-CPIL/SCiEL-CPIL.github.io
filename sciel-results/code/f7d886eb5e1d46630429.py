import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    # LCM of divisors
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    # LCM is 30030
    lcm_val = 30030
    
    best_val = float('inf')
    best_assign = None
    
    # We search for increasing W
    # Theoretical lower bound is 1 (distinct integers)
    # Upper bound: likely small
    for W in range(1, 200):
        if time.time() > deadline:
            break
            
        # For each possible start L in one period
        # We check if a valid assignment exists
        for L in range(lcm_val):
            # Check if every divisor has at least one multiple in [L, L+W]
            # and collect candidates
            candidates = []
            valid = True
            for d in divisors:
                # First multiple of d >= L
                m = ((L + d - 1) // d) * d
                if m > L + W:
                    valid = False
                    break
                # Collect all multiples in [L, L+W]
                cand = []
                while m <= L + W:
                    cand.append(m)
                    m += d
                candidates.append(cand)
            
            if not valid:
                continue
            
            # Now we have candidates for each divisor.
            # We need to pick one from each list such that all picked numbers are distinct.
            # This is a bipartite matching / SDR problem.
            # Since n=10 is small, we use backtracking.
            # To improve pruning, we process divisors with fewer candidates first.
            # But we must return results in original order.
            
            # Create list of (index, candidates)
            items = list(enumerate(candidates))
            # Sort by length of candidates (most constrained first)
            items.sort(key=lambda x: len(x[1]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                orig_idx, cands = items[idx]
                for val in cands:
                    if val not in used:
                        used.add(val)
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0):
                # Found a solution
                # Verify constraints just in case
                # 1. Distinct
                if len(set(assignment)) != n:
                    continue
                # 2. Divisibility
                ok = True
                for i in range(n):
                    if assignment[i] % divisors[i] != 0:
                        ok = False
                        break
                    if assignment[i] < 1 or assignment[i] > MAX_V:
                        ok = False
                        break
                if ok:
                    w_val = max(assignment) - min(assignment)
                    if w_val < best_val:
                        best_val = w_val
                        best_assign = assignment[:]
                    # Since we iterate W from 1, the first W found is the minimum W?
                    # Yes, because we iterate W in increasing order.
                    # However, for a fixed W, we might find a solution.
                    # Is it possible that a larger W gives a smaller range?
                    # No, the range is exactly W (max - min <= W).
                    # Wait, the range of the selected numbers is <= W.
                    # We are minimizing the range.
                    # If we find a solution in window [L, L+W], the range is <= W.
                    # If we find a solution for W=5, the range is <= 5.
                    # If we find a solution for W=4, the range is <= 4.
                    # So the first W for which a solution exists gives the minimum possible range?
                    # Not necessarily. A solution in a window of width W might have range < W.
                    # But if a solution with range R exists, it fits in a window of width R.
                    # So the minimum W for which a solution exists IS the minimum range.
                    # Thus, we can return immediately upon finding the first valid W?
                    # Yes!
                    # Because if there is a set of numbers with range R, it fits in [min, min+R].
                    # So if we check W=1, 2, ..., and find a solution for W=k,
                    # it means there is a set with range <= k.
                    # And since no solution existed for W < k, no set with range < k exists.
                    # So k is the minimum range.
                    # We can break out of loops.
                    return [int(x) for x in best_assign]
        
        # If we finished all L for this W without finding a solution, try next W
        
    # If no solution found (should not happen), return a valid fallback
    if best_assign is not None:
        return [int(x) for x in best_assign]
    
    # Fallback: just return the divisors (which is a valid solution, range is large)
    return [int(x) for x in divisors]
