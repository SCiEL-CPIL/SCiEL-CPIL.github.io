"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _get_sums(selected):
    """Compute the set of all pairwise sums (including diagonals) for a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    return sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all pair sums distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1, N]."""
    sums = _get_sums(selected)
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if adding x would create a duplicate sum
        added_sums = set()
        for a in selected:
            added_sums.add(x + a)
        added_sums.add(2 * x)
        if added_sums.isdisjoint(sums):
            return False
    return True

def _greedy_construct(order):
    """Greedy construction using the given order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _local_search_remove(selected, N, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            # Try removing selected[i]
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            if _is_valid_sidon(candidate) and _is_maximal(candidate, N):
                selected = candidate
                improved = True
                break
        if time.time() > deadline:
            break
    return selected

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    best = None
    best_size = float('inf')
    
    # Multiple random restarts with different orderings
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Generate a random ordering
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        # Greedy construction
        selected = _greedy_construct(order)
        
        if not selected:
            continue
        
        # Local search: try to remove elements
        if time.time() < deadline - 1:
            selected = _local_search_remove(selected, N, deadline)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = selected
    
    # Fallback: if no valid result, use ascending greedy
    if best is None:
        best = _greedy_construct(list(range(1, N + 1)))
    
    return best
# EVOLVE-BLOCK-END
