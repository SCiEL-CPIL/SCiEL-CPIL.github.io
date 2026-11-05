"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _conflicts(n=256):
    """Precompute conflict pairs: (i,j) where i!=j and (i*j) % (i+j) == 0"""
    conflicts = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts

def _greedy_build(conflicts, order):
    """Build a valid set by adding elements in the given order if no conflict."""
    selected = set()
    for x in order:
        if all(y not in selected for y in conflicts[x]):
            selected.add(x)
    return selected

def _local_search(conflicts, selected, max_iters=200):
    """Try to improve by swapping one element out and one in."""
    selected = set(selected)
    for _ in range(max_iters):
        improved = False
        # Try removing each element and adding a better one
        for rem in list(selected):
            temp = selected - {rem}
            # Find best candidate to add
            best_gain = -1
            best_add = None
            for cand in range(1, 257):
                if cand in temp:
                    continue
                if any(y in temp for y in conflicts[cand]):
                    continue
                # This candidate can be added; gain is 1 if we can add without removing
                # But we already removed rem, so net gain is 0 unless we find another
                # Actually, we're doing a 1-swap: remove rem, add cand
                # Net size change is 0, but might enable further additions
                # For now, just try to add cand after removing rem
                if best_gain < 0:
                    best_gain = 0
                    best_add = cand
            # Try: can we add multiple after removing rem?
            # Simpler: just try to see if removing rem lets us add more
            new_set = set(temp)
            # Greedily try to add any non-conflicting elements
            for cand in range(1, 257):
                if cand in new_set:
                    continue
                if any(y in new_set for y in conflicts[cand]):
                    continue
                new_set.add(cand)
            if len(new_set) > len(selected):
                selected = new_set
                improved = True
                break
        if not improved:
            break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    conflicts = _conflicts(256)
    
    best_set = set()
    best_size = 0
    
    # Initial: odd numbers
    odd_set = set(range(1, 257, 2))
    best_set = odd_set
    best_size = len(odd_set)
    
    # Multiple randomized greedy starts
    rng = random.Random(42)
    
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        
        # Random ordering with bias
        elements = list(range(1, 257))
        rng.shuffle(elements)
        
        # Try different strategies
        if trial % 3 == 0:
            # Pure random order
            order = elements
        elif trial % 3 == 1:
            # Sort by degree (fewer conflicts first)
            order = sorted(elements, key=lambda x: len(conflicts[x]))
        else:
            # Reverse degree sort
            order = sorted(elements, key=lambda x: -len(conflicts[x]))
        
        current = _greedy_build(conflicts, order)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
        
        # Local search on the result
        if len(current) >= best_size and trial % 5 == 0:
            improved = _local_search(conflicts, current, max_iters=50)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Final local search on best
    if time.time() < deadline - 5:
        final = _local_search(conflicts, best_set, max_iters=100)
        if len(final) > best_size:
            best_set = final
            best_size = len(final)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
