"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 256
DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _is_sidon_and_maximal(selected, N=N):
    """Check if selected is a valid Sidon set and inclusion-maximal in [1,N]."""
    if not selected:
        return False
    # Check Sidon property: all unordered pair sums a+b with a<=b must be distinct
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    
    # Check maximality: no missing element from [1,N] can be added
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Try adding x: check if x+a conflicts with any existing sum
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict:
            if (2 * x) in sums:
                conflict = True
        if not conflict:
            # Also need to check that x+x doesn't conflict (already checked via 2*x)
            # And that all x+a for a in selected are distinct from each other
            # Since x is fixed and a values are distinct, x+a values are distinct
            # So we only need to check against existing sums
            if not conflict:
                return False
    return True

def _greedy_with_order(order, N=N):
    """Greedy saturation using a given order of candidates."""
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        added_sums = [x + a for a in selected] + [2 * x]
        if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
            selected.append(x)
            sums.update(added_sums)
    return selected

def _local_improvement(selected, N=N, max_iter=200):
    """Try to reduce cardinality by removing elements and re-saturating."""
    best = list(selected)
    for _ in range(max_iter):
        improved = False
        for i in range(len(best)):
            # Try removing best[i] and see if the result is still maximal
            candidate = [x for j, x in enumerate(best) if j != i]
            if not candidate:
                continue
            # Check if candidate is still a Sidon set (it is, since subset)
            # Check if it's maximal
            if _is_sidon_and_maximal(candidate, N):
                best = candidate
                improved = True
                break
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    
    # First, try the ascending greedy as a baseline
    ascending = _greedy_with_order(range(1, N + 1))
    best = ascending
    best_size = len(best)
    
    # Try randomized greedy searches
    rng = random.Random(42)
    
    # Try many random permutations
    iterations = 0
    while time.time() - start_time < DEADLINE - 2.0:
        iterations += 1
        
        # Generate a random ordering with some structure
        # Strategy: bias towards smaller numbers first but with randomness
        order = list(range(1, N + 1))
        
        # Use a weighted shuffle: smaller numbers have higher probability of being earlier
        # This helps find compact sets
        weights = [1.0 / (i + 1) for i in range(N)]
        # Normalize
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        # Partial shuffle: swap random pairs to create diversity
        for _ in range(rng.randint(5, 30)):
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i != j:
                order[i], order[j] = order[j], order[i]
        
        result = _greedy_with_order(order)
        
        if len(result) < best_size:
            best = result
            best_size = len(result)
            # Try local improvement
            if time.time() - start_time < DEADLINE - 3.0:
                improved = _local_improvement(result)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(improved)
    
    # Final verification
    if _is_sidon_and_maximal(best, N):
        return best
    
    # Fallback: ascending greedy
    return ascending
# EVOLVE-BLOCK-END
