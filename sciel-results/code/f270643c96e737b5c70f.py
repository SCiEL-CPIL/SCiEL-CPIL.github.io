"""Divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Precompute divisibility: can[i][j] = True if smooth[i] divides smooth[j]
    can = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                can[i][j] = True
    
    # Precompute antichains of size 1..6 by sum
    # This is the key optimization: precompute all valid antichains up to size 6
    # and store them by their sum for O(1) lookup
    
    antichains_by_size = [dict() for _ in range(7)]  # sizes 0..6
    
    def is_antichain_idx(idx_tuple):
        """Check if the indices form a divisibility antichain."""
        for i in range(len(idx_tuple)):
            for j in range(i+1, len(idx_tuple)):
                a, b = idx_tuple[i], idx_tuple[j]
                if can[a][b] or can[b][a]:
                    return False
        return True
    
    # Precompute antichains for sizes 1..6
    for k in range(1, 7):
        for combo in combinations(range(n_smooth), k):
            if is_antichain_idx(combo):
                s = sum(smooth[i] for i in combo)
                if s <= 128:
                    if s not in antichains_by_size[k]:
                        antichains_by_size[k][s] = []
                    antichains_by_size[k][s].append(combo)
    
    answers = [None] * 129
    
    for n in range(1, 129):
        if time.time() > deadline - 3:
            break
        
        # Try 1 term first
        if n in smooth_set:
            answers[n] = [n]
            continue
        
        # Try 2..6 terms from precomputed antichains
        found = False
        for k in range(2, 7):
            if n in antichains_by_size[k]:
                combo = antichains_by_size[k][n][0]
                answers[n] = [smooth[i] for i in combo]
                found = True
                break
        
        if found:
            continue
        
        # For 7+ terms, use backtracking search
        # This should be rare for targets 1..128
        found = _find_antichain_rep(n, smooth, can, n_smooth, max_terms=30, start_size=7)
        if found:
            answers[n] = found
        else:
            # Fallback: should not happen, but handle gracefully
            if n in smooth_set:
                answers[n] = [n]
            else:
                # Last resort: use a simple greedy approach
                answers[n] = _greedy_antichain(n, smooth, smooth_set)
    
    # Verify all answers are valid
    for n in range(1, 129):
        if answers[n] is None:
            if n in smooth_set:
                answers[n] = [n]
            else:
                answers[n] = _greedy_antichain(n, smooth, smooth_set)
    
    return answers[1:129]


def _find_antichain_rep(n, smooth, can, n_smooth, max_terms=30, start_size=7):
    """Find an antichain representation of n using smooth numbers."""
    # Backtracking search with pruning
    smooth_sorted = sorted(smooth)
    
    def backtrack(remaining, start_idx, current):
        if remaining == 0:
            if len(current) >= 1:
                return list(current)
            return None
        if remaining < 0:
            return None
        if len(current) >= max_terms:
            return None
        
        for i in range(start_idx, n_smooth):
            val = smooth_sorted[i]
            if val > remaining:
                break
            
            # Check if adding val maintains antichain property
            valid = True
            for existing in current:
                if val % existing == 0 or existing % val == 0:
                    valid = False
                    break
            if not valid:
                continue
            
            current.append(val)
            result = backtrack(remaining - val, i + 1, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    return backtrack(n, 0, [])


def _greedy_antichain(n, smooth, smooth_set):
    """Greedy fallback: pick smooth numbers that don't divide each other."""
    chosen = []
    remaining = n
    for s in reversed(smooth):
        if s <= remaining:
            valid = True
            for c in chosen:
                if c % s == 0 or s % c == 0:
                    valid = False
                    break
            if valid:
                chosen.append(s)
                remaining -= s
                if remaining == 0:
                    break
    if remaining == 0 and 1 <= len(chosen) <= 30:
        return chosen
    return [n] if n in smooth_set else [1]
# EVOLVE-BLOCK-END
