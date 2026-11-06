"""Combined program: bitset DP for speed + multi-ordering greedy + iterative local search."""
import time
import random

MASK = (1 << 201) - 1
TARGET_BIT = 1 << 200


def _greedy_construct(order, deadline):
    """Greedy construction using bitset DP. Returns selected list."""
    selected = []
    reachable = 1  # bit 0 set: sum 0 is reachable
    for x in order:
        if time.time() > deadline:
            break
        new_reachable = reachable | (reachable << x)
        if not (new_reachable & TARGET_BIT):
            selected.append(x)
            reachable = new_reachable & MASK
    return selected


def _try_add_back(selected, deadline):
    """Iteratively try to add back excluded elements. Returns (selected, improved)."""
    selected_set = set(selected)
    all_elements = set(range(1, 101))
    excluded = sorted(all_elements - selected_set, reverse=True)
    
    # Rebuild reachable bitset
    reachable = 1
    for x in selected:
        reachable = (reachable | (reachable << x)) & MASK
    
    improved = False
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in excluded:
            if x in selected_set:
                continue
            if time.time() > deadline:
                break
            new_reachable = reachable | (reachable << x)
            if not (new_reachable & TARGET_BIT):
                selected.append(x)
                selected_set.add(x)
                reachable = new_reachable & MASK
                excluded.remove(x)
                improved = True
                changed = True
    return selected, improved


def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin for 90s limit
    
    best = []
    best_size = 0
    
    # Multiple greedy orderings
    orderings = [
        list(range(100, 0, -1)),           # descending
        list(range(1, 101)),               # ascending
        list(range(100, 0, -2)) + list(range(99, 0, -2)),  # large evens, large odds
        list(range(2, 101, 2)) + list(range(1, 100, 2)),   # evens then odds
        list(range(66, 101)) + list(range(1, 66)),         # high then low
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        sel = _greedy_construct(order, deadline)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Local search: add back excluded elements
    if time.time() < deadline:
        best, _ = _try_add_back(best, deadline)
        best_size = len(best)
    
    # Randomized restarts with perturbed orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        order = list(range(1, 101))
        rng.shuffle(order)
        # Bias toward larger numbers
        order.sort(key=lambda x: -x + rng.random() * 15)
        
        sel = _greedy_construct(order, deadline)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
            # Try to improve with local search
            if time.time() < deadline:
                best, _ = _try_add_back(best, deadline)
                best_size = len(best)
    
    # Final local search pass
    if time.time() < deadline - 2:
        best, _ = _try_add_back(best, deadline)
        best_size = len(best)
    
    # Final verification using bitset
    if best:
        reachable = 1
        for x in best:
            reachable = (reachable | (reachable << x)) & MASK
        if reachable & TARGET_BIT:
            # Should not happen, but fall back to a safe set
            best = [1]
    
    return best if best else [1]
