"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _subset_sum_bitset(elements, target=200):
    """Return a bitset (int) where bit i is 1 if sum i is achievable."""
    bits = 1  # bit 0 is set (sum 0 achievable)
    for x in elements:
        if x <= target:
            bits |= (bits << x)
    return bits

def _is_valid(elements, target=200):
    """Check if no subset of elements sums to target."""
    bits = 1
    for x in elements:
        if x <= target:
            bits |= (bits << x)
    return (bits >> target) & 1 == 0

def _greedy(order, target=200):
    """Greedy: process elements in given order, add if doesn't create target sum."""
    selected = []
    bits = 1
    for x in order:
        if x > target:
            # Can't be part of a sum to target (all positive), always safe
            selected.append(x)
            continue
        new_bits = bits | (bits << x)
        if (new_bits >> target) & 1 == 0:
            selected.append(x)
            bits = new_bits
    return selected

def _local_search(selected, target=200, max_iters=500):
    """Try to improve by swapping elements or adding/removing."""
    selected_set = set(selected)
    available = [x for x in range(1, 101) if x not in selected_set]
    
    bits = 1
    for x in selected:
        if x <= target:
            bits |= (bits << x)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try adding any available element
        for x in available:
            if x in selected_set:
                continue
            new_bits = bits | (bits << x) if x <= target else bits
            if (new_bits >> target) & 1 == 0:
                selected_set.add(x)
                available.remove(x)
                bits = new_bits
                improved = True
                break
        # Try swapping: remove one, add one
        if not improved:
            for i, x in enumerate(list(selected_set)):
                # Remove x
                new_bits = 1
                for y in selected_set:
                    if y == x:
                        continue
                    if y <= target:
                        new_bits |= (new_bits << y)
                if (new_bits >> target) & 1:
                    continue
                # Try adding each available
                for z in available:
                    if z in selected_set:
                        continue
                    final_bits = new_bits | (new_bits << z) if z <= target else new_bits
                    if (final_bits >> target) & 1 == 0:
                        selected_set.remove(x)
                        selected_set.add(z)
                        available.append(x)
                        available.remove(z)
                        bits = final_bits
                        improved = True
                        break
                if improved:
                    break
    return list(selected_set)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best = []
    best_len = 0
    
    # Try multiple orderings
    orderings = [
        list(range(1, 101)),
        list(range(100, 0, -1)),
        list(range(50, 0, -1)) + list(range(51, 101)),
        list(range(1, 51)) + list(range(100, 50, -1)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(20):
        perm = list(range(1, 101))
        rng.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy(order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best, max_iters=200)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
    
    # Ensure non-empty
    if not best:
        best = [100]
    
    return best
# EVOLVE-BLOCK-END
