"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import heapq

N = 256

# EVOLVE-BLOCK-START

def _check_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N=N):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    sel_set = set(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would create a duplicate sum
        new_sums = set()
        for a in selected:
            new_sums.add(a + x)
        new_sums.add(2 * x)
        if not sums.intersection(new_sums):
            return False  # Can be extended, not maximal
    return True

def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_complete_to_maximal(selected, N=N):
    """Try to complete a partial Sidon set to a maximal one by adding elements."""
    selected = list(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    sel_set = set(selected)
    # Try to add remaining elements in ascending order to maximize completeness
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sel_set.add(x)
            sums.update(added)
    return selected

def _local_search(selected, N=N, max_iter=10000):
    """Local search: try to reduce size by removing elements and re-completing."""
    selected = list(selected)
    best = selected[:]
    best_size = len(best)
    
    for _ in range(max_iter):
        improved = False
        for i in range(len(selected)):
            # Try removing selected[i]
            candidate = [x for j, x in enumerate(selected) if j != i]
            # Complete to maximal
            completed = _try_complete_to_maximal(candidate, N)
            if len(completed) < len(selected):
                selected = completed
                best = selected[:]
                best_size = len(best)
                improved = True
                break
        if not improved:
            break
    
    return best

def _randomized_greedy(random_order_seed, N=N):
    """Greedy with randomized ordering."""
    rng = random.Random(random_order_seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_construct(order)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best = None
    best_size = float('inf')
    
    # First, run the deterministic greedy to get a baseline
    baseline = _greedy_construct(list(range(1, N + 1)))
    best = baseline
    best_size = len(best)
    
    # Try multiple random orderings
    seed = 42
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random_order = list(range(1, N + 1))
        rng = random.Random(seed + attempts)
        rng.shuffle(random_order)
        
        candidate = _greedy_construct(random_order)
        
        # Try local search improvement
        if len(candidate) < best_size:
            improved = _local_search(candidate, N, max_iter=5000)
            if len(improved) < best_size:
                best = improved
                best_size = len(improved)
        
        # Also try starting from subsets of the baseline
        if attempts % 10 == 0 and time.time() < deadline:
            # Try removing some elements from best and re-completing
            for remove_count in [1, 2, 3]:
                if time.time() > deadline:
                    break
                for i in range(len(best)):
                    if time.time() > deadline:
                        break
                    candidate_partial = [x for j, x in enumerate(best) if j != i]
                    completed = _try_complete_to_maximal(candidate_partial, N)
                    if len(completed) < best_size:
                        best = completed
                        best_size = len(completed)
    
    # Final verification
    if best is not None:
        # Ensure it's valid and maximal
        if not _check_sidon(best):
            # Fallback to baseline
            best = _greedy_construct(list(range(1, N + 1)))
        if not _is_maximal(best, N):
            best = _try_complete_to_maximal(best, N)
    
    return best

# EVOLVE-BLOCK-END
