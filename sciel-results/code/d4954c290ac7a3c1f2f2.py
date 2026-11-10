"""Improved maximal Sidon set construction for [1,256] via randomized greedy with multiple seeds."""
import time
import random

N = 256
TIME_LIMIT = 58  # wall-clock safety margin (evaluator kills at 60s)

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all pairwise sums distinct including diagonals)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_with_order(order):
    """Build a maximal Sidon set by processing elements in the given order.
    Returns (selected, sums_set) or None if no valid set could be built."""
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        added = [x + a for a in selected] + [2 * x]
        # Check for internal collisions in added
        if len(set(added)) != len(added):
            continue
        # Check against existing sums
        if sums.intersection(added):
            continue
        selected.append(x)
        sums.update(added)
    return selected, sums

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal: no missing element from [1,N] can be added."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Try to add x
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) != len(added):
            continue
        if sums.intersection(added):
            continue
        # x can be added, so not maximal
        return False
    return True

def construct():
    start_time = time.time()
    
    # First, get the baseline greedy ascending result
    baseline_order = list(range(1, N + 1))
    baseline_selected, baseline_sums = _greedy_with_order(baseline_order)
    best_selected = baseline_selected
    best_sums = baseline_sums
    best_size = len(best_selected)
    
    # Verify baseline is maximal (it should be by construction)
    if not _is_maximal(best_selected, best_sums):
        # Shouldn't happen, but safety check
        pass
    
    # Try randomized greedy with multiple seeds
    rng = random.Random(42)
    
    # Pre-generate several random orders
    attempts = 0
    while time.time() - start_time < TIME_LIMIT - 2:
        attempts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected, sums = _greedy_with_order(order)
        if selected is None:
            continue
        
        # Check if it's maximal
        if not _is_maximal(selected, sums):
            continue
        
        size = len(selected)
        if size < best_size:
            best_selected = selected
            best_sums = sums
            best_size = size
            
            # If we've found a very small set, we can try to verify more carefully
            if best_size <= 16:
                break
    
    # Additional strategy: try orders biased toward certain ranges
    # Sometimes picking larger numbers first or with specific patterns helps
    while time.time() - start_time < TIME_LIMIT - 2:
        attempts += 1
        
        # Strategy: mix of small and large elements
        order = []
        for _ in range(5):
            # Random subset of small numbers
            small = [x for x in range(1, 33) if x not in order]
            if small:
                order.append(rng.choice(small))
            # Random subset of large numbers  
            large = [x for x in range(193, 257) if x not in order]
            if large:
                order.append(rng.choice(large))
            # Random subset of medium numbers
            med = [x for x in range(65, 193) if x not in order]
            if med:
                order.append(rng.choice(med))
        
        # Fill remaining
        remaining = [x for x in range(1, N + 1) if x not in order]
        rng.shuffle(remaining)
        order.extend(remaining)
        
        selected, sums = _greedy_with_order(order)
        if selected is None:
            continue
        
        if not _is_maximal(selected, sums):
            continue
        
        size = len(selected)
        if size < best_size:
            best_selected = selected
            best_sums = sums
            best_size = size
            if best_size <= 16:
                break
    
    # Strategy: try descending order and variations
    while time.time() - start_time < TIME_LIMIT - 2:
        attempts += 1
        
        # Descending with random perturbations
        order = list(range(N, 0, -1))
        for _ in range(rng.randint(1, 5)):
            i, j = rng.randint(0, N - 1), rng.randint(0, N - 1)
            order[i], order[j] = order[j], order[i]
        
        selected, sums = _greedy_with_order(order)
        if selected is None:
            continue
        
        if not _is_maximal(selected, sums):
            continue
        
        size = len(selected)
        if size < best_size:
            best_selected = selected
            best_sums = sums
            best_size = size
            if best_size <= 16:
                break
    
    # Try a few more targeted strategies
    while time.time() - start_time < TIME_LIMIT - 1:
        attempts += 1
        
        # Strategy: start with well-spaced elements
        spacing = rng.randint(10, 20)
        start = rng.randint(1, spacing)
        order = []
        for x in range(start, N + 1, spacing):
            order.append(x)
        # Add remaining in random order
        remaining = [x for x in range(1, N + 1) if x not in order]
        rng.shuffle(remaining)
        order.extend(remaining)
        
        selected, sums = _greedy_with_order(order)
        if selected is None:
            continue
        
        if not _is_maximal(selected, sums):
            continue
        
        size = len(selected)
        if size < best_size:
            best_selected = selected
            best_sums = sums
            best_size = size
            if best_size <= 16:
                break
    
    return best_selected
