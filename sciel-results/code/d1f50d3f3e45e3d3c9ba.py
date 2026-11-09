"""Combined program: fast greedy + local search with structural interval-based selection."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

N = 192
K = 3
MAX_ELEM = 48

def _is_valid(selected):
    """Check if selected list is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < K + 1:
        return True
    # For each element a in selected, check all triples of other elements
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        # Generate all triples from others
        for t in itertools.combinations(others, K):
            s = t[0] + t[1] + t[2]
            if s % a == 0:
                return False
    return True

def _is_valid_addition(selected, x):
    """Check if adding x to selected keeps validity."""
    # Check x against existing triples
    # x divides sum of any 3 from selected
    for t in itertools.combinations(selected, K):
        s = t[0] + t[1] + t[2]
        if s % x == 0:
            return False
    # Check existing elements against new triples involving x
    # For each a in selected, check triples from (selected \ {a}) that include x
    # i.e., triples with x and 2 from selected\{a}
    for a in selected:
        # triples: x + two others from selected\{a}
        others_without_a = [v for v in selected if v != a]
        # We need pairs from others_without_a (since x is already one element)
        for pair in itertools.combinations(others_without_a, K - 1):
            s = x + pair[0] + pair[1]
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
    """Try to improve by swapping elements out and adding better ones."""
    selected = list(selected)
    best = len(selected)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try removing one element and adding another
        for i in range(len(selected)):
            if deadline is not None and time.time() > deadline:
                break
            removed = selected[i]
            remaining = [v for j, v in enumerate(selected) if j != i]
            
            # Try adding a new element
            for x in range(N, 0, -1):
                if x == removed or x in remaining:
                    continue
                if _is_valid_addition(remaining, x):
                    new_selected = remaining + [x]
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        best = len(selected)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try single swaps that maintain size but might enable further improvements
            # Or just break
            break
    
    return selected

def construct():
    deadline = time.time() + 85  # 85s to be safe within 90s limit
    
    # Start with greedy large first
    best = greedy_large_first()
    
    # Try other greedy strategies
    alt = greedy_small_first()
    if len(alt) > len(best):
        best = alt
    
    # Try interval-based greedy with different hi_start values
    for hi_start in [192, 180, 168, 156, 144, 132, 120]:
        if time.time() > deadline:
            break
        alt = greedy_interval_based(hi_start)
        if len(alt) > len(best):
            best = alt
    
    # Try several random greedy starts
    for seed in range(50):
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
    for seed in range(50, 200):
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
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        improved = local_search_improve(best, max_iter=200, deadline=deadline)
        if len(improved) > len(best):
            best = improved
    
    return best

# EVOLVE-BLOCK-END
