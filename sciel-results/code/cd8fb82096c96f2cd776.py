"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256
SEED = 42

def _is_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal in [1, N]."""
    sel_set = set(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would create a collision
        new_sums = set()
        for a in selected:
            s = x + a
            if s in sums or s in new_sums:
                return False
            new_sums.add(s)
        s = 2 * x
        if s in sums or s in new_sums:
            return False
    return True

def _validate(selected):
    """Full validation: Sidon + maximal + distinct + in range."""
    if not selected:
        return False
    if len(set(selected)) != len(selected):
        return False
    if any(x < 1 or x > N for x in selected):
        return False
    if not _is_sidon(selected):
        return False
    if not _is_maximal(selected):
        return False
    return True

def _greedy_saturation(order):
    """Greedy saturation: pick elements in given order if they don't collide."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_blocking(order):
    """
    Greedy blocking heuristic: at each step, among available elements,
    pick the one that blocks the most future elements (i.e., creates the most
    sum collisions with other candidate elements). This makes aggressive
    blocking choices early to reduce total cardinality.
    """
    selected = []
    sums = set()
    available = list(order)
    
    while available:
        best_x = None
        best_block_count = -1
        
        for x in available:
            # Check if x can be added
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) != len(added) or sums.intersection(added):
                continue
            
            # Count how many other available elements would be blocked
            block_count = 0
            new_sums = sums.union(added)
            for y in available:
                if y == x:
                    continue
                # Check if y would be blocked by adding x
                y_added = [y + a for a in selected] + [x + y, 2 * y]
                if len(set(y_added)) != len(y_added) or new_sums.intersection(y_added):
                    block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        added = [best_x + a for a in selected[:-1]] + [2 * best_x]
        sums.update(added)
        available.remove(best_x)
    
    return selected

def _try_random_permutations(num_trials, deadline):
    """Try multiple random permutations with greedy saturation."""
    best = None
    best_len = N + 1
    
    for _ in range(num_trials):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        result = _greedy_saturation(order)
        if len(result) < best_len:
            best = result
            best_len = len(result)
    return best

def _try_blocking_heuristic(num_trials, deadline):
    """Try multiple random permutations with greedy blocking."""
    best = None
    best_len = N + 1
    
    for _ in range(num_trials):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        result = _greedy_blocking(order)
        if len(result) < best_len:
            best = result
            best_len = len(result)
    return best

def _local_search(initial, deadline):
    """
    Try to reduce cardinality via local search:
    1. Try removing elements and re-saturating
    2. Try swapping elements
    """
    current = list(initial)
    current_len = len(current)
    
    # Try removing one element and re-saturating
    for i in range(current_len):
        if time.time() > deadline:
            break
        candidate = [x for j, x in enumerate(current) if j != i]
        # Re-saturate greedily
        order = list(range(1, N + 1))
        random.shuffle(order)
        result = _greedy_saturation(order)
        # Check if result is valid and smaller
        if len(result) < current_len and _validate(result):
            current = result
            current_len = len(result)
    
    return current

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    random.seed(SEED)
    
    # First, try deterministic ascending greedy (baseline)
    baseline_order = list(range(1, N + 1))
    baseline = _greedy_saturation(baseline_order)
    best = baseline
    best_len = len(baseline)
    
    # Try random permutations with greedy saturation
    if time.time() < deadline:
        result = _try_random_permutations(500, deadline)
        if result and len(result) < best_len:
            best = result
            best_len = len(result)
    
    # Try greedy blocking heuristic
    if time.time() < deadline:
        result = _try_blocking_heuristic(100, deadline)
        if result and len(result) < best_len:
            best = result
            best_len = len(result)
    
    # Local search polish
    if time.time() < deadline and best:
        polished = _local_search(best, deadline)
        if polished and len(polished) < best_len:
            best = polished
            best_len = len(polished)
    
    # Final validation
    if best and _validate(best):
        return best
    
    # Fallback: return baseline if valid
    if baseline and _validate(baseline):
        return baseline
    
    # Last resort: return a trivial valid set
    return [1, 2, 4]
# EVOLVE-BLOCK-END
