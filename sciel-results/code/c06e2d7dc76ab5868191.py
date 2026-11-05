import time
from itertools import combinations

_START = time.time()
_DEADLINE = 85.0

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    # LCM of 1 to 22
    # 1..22: 
    # 16=2^4, 9=3^2, 5, 7, 11, 13, 17, 19, 23(not in range)
    # LCM = 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
    # 16 * 9 = 144
    # 144 * 5 = 720
    # 720 * 7 = 5040
    # 5040 * 11 = 55440
    # 55440 * 13 = 720720
    # 720720 * 17 = 12252240
    # 12252240 * 19 = 232792560
    LCM = 232792560
    weights = [LCM // k for k in range(1, 23)]
    n = 22
    
    # We want to find the largest K such that there exists a subset of size K
    # that is dissociated.
    
    def is_dissociated(indices):
        """Check if the set of indices is dissociated."""
        if not indices:
            return True
        sums = {0}
        for i in indices:
            w = weights[i]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    def backtrack(start, current, current_sums, best):
        """
        Recursive search to find a dissociated set larger than best.
        current: list of chosen indices
        current_sums: set of all subset sums of current
        """
        if not _time_ok():
            return
        if len(current) > len(best):
            best[:] = current[:]
        
        # Pruning: if we can't beat best size even by taking all remaining
        if len(current) + (n - start) <= len(best):
            return
            
        for i in range(start, n):
            if not _time_ok():
                return
            w = weights[i]
            # Check if adding i maintains dissociation
            valid = True
            new_sums = set()
            # Optimization: if current_sums is small, this is fast.
            # We can also pre-check: if w is in current_sums, it's invalid (sum w = sum w + 0)
            # Actually, the check `ns in sums` covers this if we iterate.
            # But we need to check if `s+w` collides with any existing sum.
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                # Also need to check if new_sums has internal collisions? 
                # No, if s1+w == s2+w then s1==s2, which is impossible as s1,s2 are distinct in current_sums.
                # So we just need to ensure no new sum collides with old sums.
                # Wait, what if s1+w == s2+w? Impossible.
                # What if s1+w == s3 (old sum)? Checked.
                # So `valid` is sufficient.
                
                new_total_sums = current_sums | new_sums
                current.append(i)
                backtrack(i + 1, current, new_total_sums, best)
                current.pop()
                
                # If we found a very large set, we might want to stop early?
                # The task asks to maximize. We should try to find the absolute max.
                # But if we find size 12, is it possible to have 13?
                # We can continue searching.

    # Initial best set
    best = []
    
    # Heuristic: Sort indices by weight descending? 
    # Larger weights are "harder" to form as sums of smaller weights?
    # Actually, smaller weights (larger k) are more likely to sum up to larger weights.
    # If we pick small k (large w) first, we have fewer options to collide with?
    # Let's try sorting by weight descending (k ascending).
    # Indices: 0 (k=1, w=LCM), 1 (k=2, w=LCM/2), ...
    
    # Let's try a few different orderings to be safe, but one strong one is usually enough.
    # Ordering by k ascending (weights descending) is a good heuristic for dissociated sets
    # because large weights are less likely to be represented as a sum of smaller weights
    # unless the smaller weights are very specific.
    
    # However, standard backtracking on 22 elements can be slow if the branching factor is high.
    # We can improve by trying to find a set of size K directly.
    
    # Let's estimate max size.
    # For random weights, max dissociated subset is roughly sqrt(N) or similar?
    # For structured weights like 1/k, it's different.
    # Known result: The set {1, ..., n} has a dissociated subset of size at least n/2? No.
    # Let's just run the search.
    
    # To speed up, we can use a "greedy + local search" approach first to get a good lower bound,
    # then use backtracking to prove maximality or find better.
    
    # Phase 1: Greedy to get a good initial best
    def greedy(order):
        chosen = []
        sums = {0}
        for i in order:
            w = weights[i]
            valid = True
            for s in sums:
                if (s + w) in sums:
                    valid = False
                    break
            if valid:
                chosen.append(i)
                sums.update({s + w for s in sums})
        return chosen

    # Try greedy with k ascending (weights descending)
    order_asc_k = list(range(22)) # 0..21 corresponds to k=1..22
    best = greedy(order_asc_k)
    
    # Try greedy with k descending (weights ascending)
    order_desc_k = list(range(21, -1, -1))
    cand = greedy(order_desc_k)
    if len(cand) > len(best):
        best = cand

    # Phase 2: Backtracking to improve
    # We will try to find a set larger than current best.
    # To make backtracking efficient, we sort the candidates.
    # We'll use the order that worked best for greedy, or a mixed one.
    
    # Let's use the order of indices sorted by weight descending (k ascending).
    # This tends to find large sets quickly.
    
    # We can also try to prune more aggressively.
    # If len(current) + (n - start) <= len(best): return
    
    # Let's run backtrack.
    # We need to pass the initial state.
    # We'll restart backtrack from scratch with the heuristic order.
    
    # To avoid re-exploring the same paths, we rely on the `start` index.
    
    # One optimization: If we find a set of size 12, and we know 13 is impossible (by some bound), we stop.
    # But we don't know the bound.
    
    # Let's run the backtrack.
    # We will modify best in place.
    
    # To ensure we don't time out, we check _time_ok() frequently.
    
    # We'll try to find a set of size len(best) + 1, then len(best) + 2, etc.
    # Or just run the full backtrack which updates best whenever a larger set is found.
    
    # The backtrack function above updates best whenever len(current) > len(best).
    # This is efficient because it prunes branches that cannot exceed the current best.
    
    # Let's run it.
    # We use the order k ascending (indices 0 to 21).
    backtrack(0, [], {0}, best)
    
    # If time allows, try another ordering?
    if _time_ok() and len(best) < 15: # 15 is a high guess
        # Try order k descending
        # We need a variant of backtrack that takes an order list
        pass # Skip for simplicity, one good heuristic is usually sufficient for n=22
    
    # Final verification
    if not is_dissociated(best):
        # Fallback to a known small set
        best = [0, 1, 3, 7] # k=1,2,4,8 -> powers of 2 are always dissociated
        
    return [i + 1 for i in best]
