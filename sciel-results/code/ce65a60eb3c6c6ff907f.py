"""Combined program: fast greedy + local search with structural interval-based selection."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

N = 192
K = 3
MAX_ELEM = 48

def _precompute_pair_sums(selected):
    """Precompute all pair sums from selected list."""
    n = len(selected)
    pair_sums = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_sums[(i, j)] = selected[i] + selected[j]
    return pair_sums

def _is_valid_addition(selected, x):
    """Check if adding x to selected keeps validity. Optimized."""
    n = len(selected)
    if n < K:
        return True
    
    # Check x against existing triples: x divides sum of any 3 from selected
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = selected[i] + selected[j] + selected[k]
                if s % x == 0:
                    return False
    
    # Check existing elements against new triples involving x
    # For each a in selected, check triples (x, b, c) where b,c from selected\{a}
    for i, a in enumerate(selected):
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                s = x + selected[j] + selected[k]
                if s % a == 0:
                    return False
    return True

def _is_valid(selected):
    """Check if selected list is valid."""
    n = len(selected)
    if n < K + 1:
        return True
    for i in range(n):
        a = selected[i]
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                for m in range(k + 1, n):
                    if m == i:
                        continue
                    s = selected[j] + selected[k] + selected[m]
                    if s % a == 0:
                        return False
    return True

def greedy_large_first():
    """Greedy: try numbers from 192 down to 1."""
    selected = []
    for x in range(N, 0, -1):
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def greedy_small_first():
    """Greedy: try numbers from 1 up to 192."""
    selected = []
    for x in range(1, N + 1):
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def greedy_random(seed):
    """Greedy with random order."""
    rng = random.Random(seed)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    selected = []
    for x in candidates:
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def greedy_interval_based(hi_start):
    """Greedy starting from high range, then filling in lower range."""
    order = list(range(hi_start, N + 1)) + list(range(hi_start - 1, 0, -1))
    selected = []
    for x in order:
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def local_search_improve(selected, max_iter=200, deadline=None):
    """Try to improve by various swap operations."""
    selected = list(selected)
    best = len(selected)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try removing one element and adding two (net +1)
        for i in range(len(selected)):
            if deadline is not None and time.time() > deadline:
                break
            removed = selected[i]
            remaining = [v for j, v in enumerate(selected) if j != i]
            
            # Try adding two new elements
            for x in range(N, 0, -1):
                if x == removed or x in remaining:
                    continue
                if not _is_valid_addition(remaining, x):
                    continue
                temp = remaining + [x]
                for y in range(N, 0, -1):
                    if y == removed or y == x or y in remaining:
                        continue
                    if _is_valid_addition(temp, y):
                        new_selected = temp + [y]
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            best = len(selected)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try removing two and adding one (net -1) for restructuring
        if len(selected) > 2:
            for i in range(len(selected)):
                if deadline is not None and time.time() > deadline:
                    break
                for j in range(i + 1, len(selected)):
                    if deadline is not None and time.time() > deadline:
                        break
                    removed1 = selected[i]
                    removed2 = selected[j]
                    remaining = [v for k, v in enumerate(selected) if k != i and k != j]
                    
                    for x in range(N, 0, -1):
                        if x == removed1 or x == removed2 or x in remaining:
                            continue
                        if _is_valid_addition(remaining, x):
                            new_selected = remaining + [x]
                            # This reduces size by 1, but might enable future growth
                            # Only accept if it leads to improvement after more search
                            # For now, skip this as it's complex
                            pass
                    # Skip complex 2-for-1 for now
        
        # Try single swaps (remove 1, add 1) for rearranging
        for i in range(len(selected)):
            if deadline is not None and time.time() > deadline:
                break
            removed = selected[i]
            remaining = [v for j, v in enumerate(selected) if j != i]
            
            for x in range(N, 0, -1):
                if x == removed or x in remaining:
                    continue
                if _is_valid_addition(remaining, x):
                    new_selected = remaining + [x]
                    # Same size, but different composition - might enable future growth
                    # Only accept with some probability or if it seems promising
                    # For simplicity, skip
                    pass
            # Skip single swaps for now
        
        if not improved:
            break
    
    return selected

def construct():
    deadline = time.time() + 82  # 82s to be safe within 90s limit
    
    # Start with greedy large first
    best = greedy_large_first()
    
    # Try other greedy strategies
    alt = greedy_small_first()
    if len(alt) > len(best):
        best = alt
    
    # Try interval-based greedy with different hi_start values
    for hi_start in [192, 180, 168, 156, 144, 132, 120, 108, 96, 84, 72, 60, 48, 36, 24, 12]:
        if time.time() > deadline:
            break
        alt = greedy_interval_based(hi_start)
        if len(alt) > len(best):
            best = alt
    
    # Try several random greedy starts
    for seed in range(100):
        if time.time() > deadline:
            break
        alt = greedy_random(seed)
        if len(alt) > len(best):
            best = alt
    
    # Local search improvement
    improved = local_search_improve(best, max_iter=100, deadline=deadline)
    if len(improved) > len(best):
        best = improved
    
    # More random restarts if time permits
    for seed in range(100, 300):
        if time.time() > deadline:
            break
        alt = greedy_random(seed)
        if len(alt) > len(best):
            best = alt
            # Try local search on the new best
            improved = local_search_improve(best, max_iter=50, deadline=deadline)
            if len(improved) > len(best):
                best = improved
    
    # Final local search polish
    if time.time() < deadline - 3:
        improved = local_search_improve(best, max_iter=200, deadline=deadline)
        if len(improved) > len(best):
            best = improved
    
    return best

# EVOLVE-BLOCK-END
