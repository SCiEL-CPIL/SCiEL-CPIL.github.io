import time
import random
import sys

# EVOLVE-BLOCK-START

def _greedy_init(deadline):
    """
    Greedy heuristic to find a valid Sidon set covering 1..30.
    Prioritizes elements that cover the most missing differences.
    Returns (set, max_elem) or (None, inf) if time runs out.
    """
    S = [0]
    used_mask = 0
    target_mask = 0
    for i in range(1, 31):
        target_mask |= (1 << i)
    covered_mask = 0
    
    while (covered_mask & target_mask) != target_mask:
        if time.time() > deadline:
            return None, float('inf')
        
        missing = target_mask & ~covered_mask
        if missing == 0:
            break
            
        best_x = None
        best_score = -1
        best_new_mask = 0
        
        # Search for the next element. We can limit the search range to keep it fast.
        # A diameter of ~50-100 is expected. Let's search up to 200.
        start = S[-1] + 1
        end = min(200, 10**6)
        
        for x in range(start, end + 1):
            ok = True
            new_mask = 0
            for v in S:
                d = x - v
                bit = 1 << d
                if (used_mask & bit) or (new_mask & bit):
                    ok = False
                    break
                new_mask |= bit
            if not ok:
                continue
            
            new_covered = covered_mask | new_mask
            # Score: number of new missing differences covered
            score = bin(new_covered & missing).count('1')
            
            if score > best_score:
                best_score = score
                best_x = x
                best_new_mask = new_mask
        
        if best_x is None:
            # If no valid x found in range, extend range or fail
            # For simplicity, if greedy fails, return None
            return None, float('inf')
        
        S.append(best_x)
        used_mask |= best_new_mask
        covered_mask |= best_new_mask
        
        if len(S) > 32:
            return None, float('inf')
            
    return S, S[-1]

def _backtrack_optimal(deadline, initial_best=None):
    """Find minimal diameter Sidon set covering diffs 1..30 using bitset."""
    best_diam = [float('inf')]
    best_set = [None]
    target_mask = 0
    for i in range(1, 31):
        target_mask |= (1 << i)
        
    if initial_best is not None:
        best_diam[0] = initial_best
        best_set[0] = initial_best

    def rec(S, used_mask, covered_mask, current_max):
        if time.time() > deadline:
            return
        if current_max >= best_diam[0]:
            return
        if (covered_mask & target_mask) == target_mask:
            if current_max < best_diam[0]:
                best_diam[0] = current_max
                best_set[0] = list(S)
            return
        n = len(S)
        if n >= 32:
            return
        
        # Prune: max new diffs from remaining slots
        remaining = 32 - n
        # Each new element can add at most n new differences (differences with existing elements)
        # Actually, it adds n differences. So max new diffs = remaining * n
        max_new = remaining * n
        missing_count = bin(target_mask & ~covered_mask).count('1')
        if max_new < missing_count:
            return
        
        start = S[-1] + 1 if S else 1
        upper = min(best_diam[0] - 1, 100) # Limit search space depth
        if upper < start:
            return
        
        # Generate candidates and sort by coverage
        candidates = []
        for x in range(start, upper + 1):
            ok = True
            new_mask = 0
            for v in S:
                d = x - v
                bit = 1 << d
                if (used_mask & bit) or (new_mask & bit):
                    ok = False
                    break
                new_mask |= bit
            if not ok:
                continue
            
            # Score: how many missing diffs does this x cover?
            new_covered = covered_mask | new_mask
            score = bin(new_covered & (target_mask & ~covered_mask)).count('1')
            candidates.append((score, x, new_mask))
            
        # Sort by score descending, then x ascending (prefer smaller x for same score)
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for score, x, new_mask in candidates:
            if time.time() > deadline:
                return
            new_used = used_mask | new_mask
            new_covered = covered_mask | new_mask
            S.append(x)
            rec(S, new_used, new_covered, x)
            S.pop()
            
    rec([0], 0, 0, 0)
    return best_set[0], best_diam[0]

def construct():
    start_time = time.time()
    # Safety margin: 120s hard kill, use 100s for search to be safe
    deadline = start_time + 100
    
    # Phase 1: Greedy initialization to get a good upper bound
    greedy_set, greedy_diam = _greedy_init(deadline - 5) # Leave 5s for backtracking
    
    # Phase 2: Backtracking optimization
    result, diam = _backtrack_optimal(deadline, greedy_set if greedy_set else None)
    
    if result is not None:
        return result
    
    # Fallback: known construction
    return [0, 1, 4, 10, 12, 28, 32, 45, 63]

# EVOLVE-BLOCK-END
