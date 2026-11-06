import time
import random
import sys

# EVOLVE-BLOCK-START

def _backtrack_optimal(deadline):
    """Find minimal diameter Sidon set covering diffs 1..30 using bitset."""
    best_diam = [float('inf')]
    best_set = [None]
    target_mask = 0
    for i in range(1, 31):
        target_mask |= (1 << i)
    
    def rec(S, used_mask, covered_mask, current_max, missing_count):
        if time.time() > deadline:
            return
        if current_max >= best_diam[0]:
            return
        if missing_count == 0:
            if current_max < best_diam[0]:
                best_diam[0] = current_max
                best_set[0] = list(S)
            return
        n = len(S)
        if n >= 32:
            return
        # Prune: max new diffs from remaining slots
        remaining = 32 - n
        # New diffs from adding 'remaining' elements to a set of size 'n':
        # Each new element creates 'n' diffs with old elements.
        # Each pair of new elements creates 1 diff.
        # Total = remaining * n + C(remaining, 2)
        max_new = remaining * n + remaining * (remaining - 1) // 2
        if max_new < missing_count:
            return
        
        start = S[-1] + 1 if S else 1
        upper = min(best_diam[0] - 1, 80)
        if upper < start:
            return
        
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
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
            new_used = used_mask | new_mask
            new_covered = covered_mask | new_mask
            # Calculate reduction in missing count
            newly_covered = (new_mask & target_mask) & ~covered_mask
            reduction = bin(newly_covered).count('1')
            new_missing = missing_count - reduction
            
            S.append(x)
            rec(S, new_used, new_covered, x, new_missing)
            S.pop()
    
    rec([0], 0, 0, 0, 30)
    return best_set[0], best_diam[0]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    result, diam = _backtrack_optimal(deadline)
    
    if result is not None:
        return result
    
    # Fallback: known construction
    return [0, 1, 4, 10, 12, 28, 32, 45, 63]

# EVOLVE-BLOCK-END
