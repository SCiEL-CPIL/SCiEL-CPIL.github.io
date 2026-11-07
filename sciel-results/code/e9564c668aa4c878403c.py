import time
import sys
from collections import deque

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def _find_matching(candidates):
    """Try to find a perfect matching via backtracking. Returns assignment or None."""
    assignment = [None] * N
    used = set()
    
    # Sort by number of candidates for better pruning
    indices = sorted(range(N), key=lambda i: len(candidates[i]))
    
    def backtrack(idx):
        if idx == len(indices):
            return True
        i = indices[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assignment[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assignment[i] = None
        return False
    
    if backtrack(0):
        return [int(assignment[i]) for i in range(N)]
    return None

def _check_window(L, S):
    """Check if window [L, L+S] has a valid distinct assignment. Returns result or None."""
    if L < 1 or L + S > MAXV:
        return None
    
    candidates = []
    for d in DIVISORS:
        k = (L + d - 1) // d
        m = k * d
        cands = []
        while m <= L + S:
            cands.append(m)
            m += d
        if not cands:
            return None
        candidates.append(cands)
    
    return _find_matching(candidates)

def _generate_candidate_Ls(S, max_L):
    """Generate candidate L values near multiples of divisors within [1, max_L]."""
    cand_set = set()
    
    # For each divisor, the multiples that could be the "anchor" in a window of span S
    # If multiple m of d is in [L, L+S], then L is in [m-S, m]
    # So L must be within S of some multiple of each divisor.
    # Candidate L values are near multiples of divisors.
    
    for d in DIVISORS:
        # Multiples of d in range [1, max_L + S]
        first_k = max(1, 1)  # start from 1
        m = d
        while m <= max_L + S:
            # L can be in [m-S, m] intersect [1, max_L]
            lo = max(1, m - S)
            hi = min(max_L, m)
            # Add a few candidate L values: m, m-1, ..., m-S (but bounded)
            # To keep it manageable, add L = m - j for j in 0..min(S, 10)
            for j in range(min(S, 15) + 1):
                L_val = m - j
                if 1 <= L_val <= max_L:
                    cand_set.add(L_val)
            m += d
    
    return sorted(cand_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_result = None
    best_span = float('inf')
    
    # Phase 1: Targeted search over candidate L values for increasing S
    max_S = 200
    max_L = 1000000
    
    for S in range(1, max_S + 1):
        if time.time() > deadline:
            break
        
        # Generate candidate L values near multiples of divisors
        # Limit the number of candidates to keep it fast
        cand_Ls = _generate_candidate_Ls(S, max_L)
        
        # If too many candidates, sample them
        if len(cand_Ls) > 5000:
            # Sample evenly
            step = len(cand_Ls) // 5000
            cand_Ls = cand_Ls[::step]
        
        for L in cand_Ls:
            if time.time() > deadline:
                break
            
            result = _check_window(L, S)
            if result is not None:
                span = max(result) - min(result)
                if span <= S:
                    # Found a valid solution with span <= S
                    # Since we're searching in increasing S, this is optimal
                    return result
                else:
                    # Update best if better
                    if span < best_span:
                        best_span = span
                        best_result = result
    
    # Phase 2: If we have a best result, try to polish it
    if best_result is not None and time.time() < deadline - 10:
        # Local search: try shifting the window slightly
        cur_L = min(best_result)
        cur_S = max(best_result) - min(best_result)
        
        # Try to find a tighter window by adjusting L
        for delta_L in range(-50, 51):
            if time.time() > deadline:
                break
            new_L = cur_L + delta_L
            if new_L < 1:
                continue
            # Try span from 1 up to cur_S
            for new_S in range(1, cur_S + 1):
                if time.time() > deadline:
                    break
                result = _check_window(new_L, new_S)
                if result is not None:
                    span = max(result) - min(result)
                    if span < best_span:
                        best_span = span
                        best_result = result
                    break  # Found a valid one for this new_L, move to next delta_L
    
    if best_result is not None:
        return best_result
    
    # Fallback: return divisors themselves
    return DIVISORS[:]
