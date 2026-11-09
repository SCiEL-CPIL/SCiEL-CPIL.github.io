"""Combined program: fast greedy + local search with structural interval-based selection."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

N = 192
K = 3
MAX_ELEM = 48

def _can_add(x, selected):
    """
    Check if x can be added to selected without violating constraints.
    Optimized for speed: 
    1. Check if x divides sum of any 3 from selected.
    2. Check if any a in selected divides sum of any 3 from (selected U {x}) \ {a}.
       This is equivalent to checking if a divides (x + b + c) for any b, c in selected \ {a}.
    """
    m = len(selected)
    
    # 1. Check: does x divide sum of any 3 from selected?
    if m >= 3:
        # Optimization: If x is large, it's less likely to divide sums. 
        # But we must check all triples.
        # Using itertools.combinations is generally fast enough for m <= 48.
        for i in range(m):
            a = selected[i]
            for j in range(i + 1, m):
                b = selected[j]
                for k in range(j + 1, m):
                    c = selected[k]
                    if (a + b + c) % x == 0:
                        return False
    
    # 2. Check: for each a in selected, does a divide sum of any 3 from (selected U {x}) \ {a}?
    # The new triples involving x are of the form (x, b, c) where b, c are in selected \ {a}.
    # So we check if (x + b + c) % a == 0 for any b, c in selected \ {a}.
    for i, a in enumerate(selected):
        # We need to check pairs (b, c) from selected excluding a
        # If m is small, this is fast. If m is large, this is the bottleneck.
        # We can iterate over all pairs in selected and skip if either is a.
        for j in range(m):
            if j == i:
                continue
            b = selected[j]
            for k in range(j + 1, m):
                if k == i:
                    continue
                c = selected[k]
                if (x + b + c) % a == 0:
                    return False
                    
    return True

def _greedy_build(order, max_elem=MAX_ELEM):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= max_elem:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit, deadline=None):
    """
    Two-phase local search: 
    1. Try pure additions.
    2. Try single swaps.
    3. Try two-swap moves (remove 2, add 2) to escape local optima.
    """
    start = time.time()
    best = list(selected)
    
    def _time_ok():
        if deadline is not None:
            return time.time() < deadline
        return time.time() - start < time_limit
    
    while _time_ok():
        improved = False
        # Candidates are numbers not in best, sorted descending (larger numbers are "safer" to add)
        candidates = [x for x in range(N, 0, -1) if x not in best]
        
        # Phase 1: Try pure additions
        for x in candidates:
            if not _time_ok():
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try single swaps (remove one, add one)
        # We try to remove the "most restrictive" element first? 
        # Or just iterate. Removing a small element might allow adding a larger one.
        # Let's try removing elements in order of increasing value (smaller elements are more likely to divide sums).
        indices = list(range(len(best)))
        # Sort indices by the value of the element they point to (ascending)
        indices.sort(key=lambda i: best[i])
        
        for i in indices:
            if not _time_ok():
                break
            removed = best[i]
            remaining = best[:i] + best[i+1:]
            
            for x in candidates:
                if x == removed:
                    continue
                if _can_add(x, remaining):
                    best = remaining + [x]
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 3: Try two-swap moves (remove two, add two)
        # This is expensive, so we limit the search space.
        # We try removing pairs of small elements and adding pairs of large elements.
        # To keep it fast, we only try a limited number of pairs.
        if len(best) >= 2:
            # Generate a limited set of removal pairs
            # Sort by value ascending to prioritize removing "bad" small numbers
            sorted_indices = sorted(range(len(best)), key=lambda i: best[i])
            # Try first 10 pairs from the sorted list
            for p in range(10):
                if not _time_ok():
                    break
                for q in range(p + 1, min(p + 10, len(sorted_indices))):
                    if not _time_ok():
                        break
                    i, j = sorted_indices[p], sorted_indices[q]
                    if i == j:
                        continue
                    removed1, removed2 = best[i], best[j]
                    # Create remaining list excluding i and j
                    # Note: indices might shift if i > j, so we filter by value or careful index handling
                    # Simpler: filter by value if values are distinct (they are)
                    remaining = [v for v in best if v != removed1 and v != removed2]
                    
                    # Try adding two new elements from candidates
                    # Limit to top 15 candidates for speed
                    for x in candidates[:15]:
                        if not _time_ok():
                            break
                        if x == removed1 or x == removed2:
                            continue
                        if _can_add(x, remaining):
                            temp1 = remaining + [x]
                            for y in candidates[:15]:
                                if not _time_ok():
                                    break
                                if y == removed1 or y == removed2 or y == x:
                                    continue
                                if _can_add(y, temp1):
                                    best = temp1 + [y]
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Strategy 1: Greedy from large to small (Strongest structural heuristic)
    best = _greedy_build(list(range(N, 0, -1)))
    
    # Strategy 2: Greedy from small to large
    alt = _greedy_build(list(range(1, N + 1)))
    if len(alt) > len(best):
        best = alt
    
    # Strategy 3: Interval-based greedy with various hi_start values
    # Start with high numbers, then fill in lower ones
    for hi_start in [192, 180, 168, 156, 144, 132, 120, 108, 96]:
        if time.time() > deadline - 20:
            break
        order = list(range(hi_start, N + 1)) + list(range(hi_start - 1, 0, -1))
        alt = _greedy_build(order)
        if len(alt) > len(best):
            best = alt
    
    # Strategy 4: Randomized greedy with multiple seeds
    # Focus on random orders that still prioritize larger numbers slightly
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        # Bias towards larger numbers by sorting with random noise
        # Or just shuffle
        alt = _greedy_build(candidates)
        if len(alt) > len(best):
            best = alt
    
    # Local search improvement on the best greedy result
    if time.time() < deadline - 5:
        remaining = min(deadline - time.time() - 2, 30.0)
        improved = _local_search(best, remaining, deadline=deadline)
        if len(improved) > len(best):
            best = improved
    
    # More random restarts with local search if time permits
    for seed in range(50, 150):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        
        # Only run local search if the greedy result is close to the best
        if len(alt) >= len(best) - 2:
            remaining = min(deadline - time.time() - 1, 10.0)
            if remaining > 1.0:
                alt = _local_search(alt, remaining, deadline=deadline)
                if len(alt) > len(best):
                    best = alt
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = min(deadline - time.time() - 1, 10.0)
        if remaining > 0.5:
            improved = _local_search(best, remaining, deadline=deadline)
            if len(improved) > len(best):
                best = improved
    
    return best

# EVOLVE-BLOCK-END
