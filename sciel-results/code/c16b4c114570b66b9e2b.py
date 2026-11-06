"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    n_smooth = len(smooth_numbers)
    smooth_set = set(smooth_numbers)
    
    # Precompute conflict matrix: conflict[i][j] = True if smooth_numbers[i] divides smooth_numbers[j] or vice versa
    # For antichain: we need NO pair to have conflict
    conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_numbers[i], smooth_numbers[j]
                if a % b == 0 or b % a == 0:
                    conflict[i][j] = True
    
    # Precompute for each index i, a bitmask of conflicting indices
    # conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(n_smooth):
            if i != j and conflict[i][j]:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # For each target n, find minimum k such that there exist k distinct smooth numbers
    # <= n that sum to n and form an antichain.
    
    answers = []
    
    # Precompute: for each possible subset size k (1 to max_k), store valid antichain subsets
    # as (sum, bitmask, tuple_of_values)
    # But we'll do it per-target for efficiency
    
    # For each n, try k=1,2,3,4,5,...
    # Use bitmask-based antichain check for speed
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Fallback for remaining targets
            for m in range(n, 129):
                if m in smooth_set:
                    answers.append([m])
                else:
                    # Simple fallback
                    answers.append([1, m - 1] if (m - 1) in smooth_set and (m-1) % 1 != 0 else [m])
            break
        
        candidates = [x for x in smooth_numbers if x <= n]
        candidates_set = set(candidates)
        cand_indices = [smooth_numbers.index(x) for x in candidates]
        
        found = None
        
        # k=1
        if n in candidates_set:
            found = [n]
        else:
            # k=2
            found = None
            for a in candidates:
                b = n - a
                if b <= a:
                    continue
                if b in candidates_set:
                    # Check antichain: b % a != 0
                    if b % a != 0:
                        found = [a, b]
                        break
            
            if found is None:
                # k=3
                found = None
                for i in range(len(candidates)):
                    a = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        b = candidates[j]
                        c = n - a - b
                        if c <= b:
                            continue
                        if c in candidates_set:
                            # Check antichain
                            if b % a != 0 and c % a != 0 and c % b != 0:
                                found = [a, b, c]
                                break
                    if found is not None:
                        break
            
            if found is None:
                # k=4 and above: use bitmask-based search
                found = _find_min_antician(n, candidates, candidates_set, cand_indices, conflict_mask, smooth_numbers, start_time, timeout)
        
        if found is None:
            # Fallback
            if n in smooth_set:
                found = [n]
            else:
                found = [1, n - 1]
        
        answers.append(found)
    
    return answers

def _find_min_antician(n, candidates, candidates_set, cand_indices, conflict_mask, smooth_numbers, start_time, timeout):
    """Find minimum-k antichain representation using bitmask DP."""
    # Try k=4, 5, 6, ...
    max_k = min(len(candidates), 15)
    
    for k in range(4, max_k + 1):
        if time.time() - start_time > timeout - 3:
            break
        
        # Use backtracking with bitmask pruning
        result = _backtrack_bitmask(n, k, candidates, cand_indices, conflict_mask, smooth_numbers, start_time, timeout)
        if result is not None:
            return result
    
    # Last resort: try larger k
    for k in range(max_k + 1, min(len(candidates), 25) + 1):
        if time.time() - start_time > timeout - 2:
            break
        result = _backtrack_bitmask(n, k, candidates, cand_indices, conflict_mask, smooth_numbers, start_time, timeout)
        if result is not None:
            return result
    
    return None

def _backtrack_bitmask(remaining, k, candidates, cand_indices, conflict_mask, smooth_numbers, start_time, timeout):
    """Backtracking with bitmask-based antichain check."""
    if k == 0:
        return [] if remaining == 0 else None
    
    if remaining <= 0 or len(candidates) < k:
        return None
    
    # Pruning: sum of k smallest must be <= remaining
    sorted_cands = sorted(candidates)
    if sum(sorted_cands[:k]) > remaining:
        return None
    
    # Pruning: sum of k largest must be >= remaining
    if sum(sorted_cands[-k:]) < remaining:
        return None
    
    for i in range(len(candidates)):
        a = candidates[i]
        if a > remaining:
            continue
        
        # Check conflict with already chosen (we track via the recursion structure)
        # In this version, we pass candidates[i+1:] so no duplicates
        # But we still need to check antichain property
        # Since we're building incrementally, we need to track chosen elements
        # Let's use a different approach: pass a list of chosen indices
        
        # Actually, let's restructure: use a global chosen list
        pass
    
    # Simpler approach: just do the search with explicit antichain check
    # but with better pruning
    return _backtrack_simple(remaining, k, candidates, start_time, timeout)

def _backtrack_simple(remaining, k, candidates, start_time, timeout):
    """Simple backtracking with pruning."""
    if k == 0:
        return [] if remaining == 0 else None
    
    if remaining <= 0 or len(candidates) < k:
        return None
    
    if time.time() - start_time > timeout - 2:
        return None
    
    # Pruning
    sorted_cands = sorted(candidates)
    if sum(sorted_cands[:k]) > remaining:
        return None
    if sum(sorted_cands[-k:]) < remaining:
        return None
    
    # Try each candidate
    for i in range(len(candidates)):
        a = candidates[i]
        if a > remaining:
            continue
        
        # Check if a conflicts with any other candidate in the remaining search space
        # that might also be chosen - this is complex, so just do simple check
        # and rely on the final antichain verification
        
        # Recurse
        new_candidates = candidates[i+1:]
        result = _backtrack_simple(remaining - a, k - 1, new_candidates, start_time, timeout)
        if result is not None:
            # Verify antichain
            full = [a] + result
            ok = True
            for i2 in range(len(full)):
                for j2 in range(i2 + 1, len(full)):
                    x, y = full[i2], full[j2]
                    if x < y:
                        if y % x == 0:
                            ok = False
                            break
                    else:
                        if x % y == 0:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                return full
    
    return None
# EVOLVE-BLOCK-END
