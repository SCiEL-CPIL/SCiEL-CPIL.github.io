"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_200_free(selected):
    """Check if no subset of selected sums to 200."""
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return False
        reachable |= new
    return True

def _greedy_construct(order, target=200):
    """Greedy construction: add elements in given order if safe."""
    selected = []
    reachable = {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= target}
        if target not in new:
            selected.append(x)
            reachable |= new
    return selected

def _local_search(selected, all_elements, target=200):
    """Try to add more elements to selected."""
    selected_set = set(selected)
    improved = True
    while improved:
        improved = False
        for x in all_elements:
            if x not in selected_set:
                candidate = selected + [x]
                if _is_200_free(candidate):
                    selected = candidate
                    selected_set.add(x)
                    improved = True
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second deadline to be safe
    
    all_elements = list(range(1, 101))
    
    # First, run the original greedy (descending order)
    best = _greedy_construct(sorted(all_elements, reverse=True))
    best_size = len(best)
    
    # Try ascending order
    candidate = _greedy_construct(all_elements)
    if len(candidate) > best_size:
        best = candidate
        best_size = len(candidate)
    
    # Try random restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Random order
        order = all_elements.copy()
        random.shuffle(order)
        candidate = _greedy_construct(order)
        
        # Local search
        candidate = _local_search(candidate, all_elements)
        
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
            if best_size == 100:
                break
        elif len(candidate) == best_size:
            # Try local search even if same size, might find better
            pass
    
    # Final local search on best
    if time.time() < deadline - 2:
        best = _local_search(best, all_elements)
    
    return best
# EVOLVE-BLOCK-END
