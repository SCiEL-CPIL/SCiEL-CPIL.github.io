"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, n=N):
    """Check if selected is inclusion-maximal in [1, n]."""
    selected_set = set(selected)
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        # Try adding x
        candidate = list(selected) + [x]
        if _is_sidon(candidate):
            return False
    return True

def _greedy_sidon(order):
    """Build a Sidon set by greedily adding elements in the given order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_reduce(selected):
    """Try to remove elements from a maximal Sidon set while preserving maximality."""
    if not _is_maximal(selected):
        return selected
    
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            # Check if candidate is still Sidon (it should be, since it's a subset)
            # Check if candidate is still maximal
            if _is_maximal(candidate):
                selected = candidate
                improved = True
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    best = None
    best_size = N + 1
    
    # Try multiple random orderings with local reduction
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Create a random permutation of [1, 256]
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        # Build a Sidon set using this order
        selected = _greedy_sidon(order)
        
        # Try to reduce while maintaining maximality
        reduced = _try_reduce(selected)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best = reduced
            
            # Early termination if we hit a good size
            if best_size <= 16:
                break
    
    # Also try ascending order as a baseline
    if best is None or len(_try_reduce(_greedy_sidon(list(range(1, N+1))))) < best_size:
        baseline = _try_reduce(_greedy_sidon(list(range(1, N+1))))
        if len(baseline) < best_size:
            best = baseline
    
    if best is None:
        # Fallback: simple ascending greedy
        best = _greedy_sidon(list(range(1, N + 1)))
    
    return best
# EVOLVE-BLOCK-END
