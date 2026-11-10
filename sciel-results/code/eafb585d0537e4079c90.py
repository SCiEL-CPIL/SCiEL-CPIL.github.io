"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _score(chosen):
    return sum(1.0 / (512 - a) for a in chosen)

def _is_coprime_to_all(a, chosen):
    for b in chosen:
        if gcd(a, b) != 1:
            return False
    return True

def _greedy_from_high():
    chosen = []
    for a in range(511, 0, -1):
        if _is_coprime_to_all(a, chosen):
            chosen.append(a)
    return chosen

def _local_search(chosen, deadline):
    """Try to improve by removing one element and re-greeding."""
    best = list(chosen)
    best_score = _score(best)
    n = len(best)
    # Try removing each element
    for i in range(n):
        if time.time() > deadline:
            break
        # Remove best[i]
        remaining = [x for j, x in enumerate(best) if j != i]
        # Re-greedily add elements from high to low that are coprime to remaining
        new_set = set(remaining)
        new_list = list(remaining)
        for a in range(511, 0, -1):
            if a in new_set:
                continue
            if _is_coprime_to_all(a, new_list):
                new_list.append(a)
                new_set.add(a)
        new_score = _score(new_list)
        if new_score > best_score:
            best = new_list
            best_score = new_score
            n = len(best)
            # Restart the loop with the new best
            i = -1
    return best

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    # Initial greedy
    chosen = _greedy_from_high()
    
    # Local search: try removing elements and re-greeding
    chosen = _local_search(chosen, deadline)
    
    return chosen
# EVOLVE-BLOCK-END
