"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    # Check existing elements against new element
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        # Add x to others for checking
        others_with_x = others + [x]
        for triple in combinations(others_with_x, 3):
            if sum(triple) % a == 0:
                return False
    # Check new element against existing elements
    others_for_x = list(selected)
    for triple in combinations(others_for_x, 3):
        if sum(triple) % x == 0:
            return False
    return True

def _greedy(order):
    """Greedy construction with given order."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Local search: try to add more elements or swap."""
    selected = list(selected)
    n = len(selected)
    
    # Try to add elements
    candidates = [x for x in range(1, 193) if x not in selected]
    for x in candidates:
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    
    # Try swaps: remove one, add two
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        for j in range(len(selected)):
            if i == j:
                continue
            # Try removing selected[i] and adding a new element
            remaining = [selected[k] for k in range(len(selected)) if k != i]
            for x in range(1, 193):
                if x in remaining or x == selected[i]:
                    continue
                if _can_add(remaining, x):
                    new_selected = remaining + [x]
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Try multiple greedy orderings
    orders = [
        list(range(192, 0, -1)),  # descending
        list(range(1, 193)),      # ascending
    ]
    
    # Add some random orders
    for _ in range(10):
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy(order)
        if len(result) > len(best):
            best = result
    
    # Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy(list(range(192, 0, -1)))
    
    return best
# EVOLVE-BLOCK-END
