"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_3ap_with(selected_set, x):
    """Check if adding x to selected_set creates a 3-term AP. O(n) per check."""
    for a in selected_set:
        # Check if a, x, b form AP where b = 2x - a
        b = 2 * x - a
        if b != a and b != x and b in selected_set:
            return True
        # Check if x, a, b form AP where b = 2a - x
        b = 2 * a - x
        if b != a and b != x and b in selected_set:
            return True
        # Check if a, b, x form AP where b = (a + x) / 2
        if (a + x) % 2 == 0:
            b = (a + x) // 2
            if b != a and b != x and b in selected_set:
                return True
    return False

def _reciprocal_sum(selected):
    """Compute sum of reciprocals as float."""
    return sum(1.0 / x for x in selected)

def _greedy_ascending():
    """Greedy: iterate x from 1 to 256, add if no 3-AP is created."""
    selected = set()
    for x in range(1, 257):
        if not _has_3ap_with(selected, x):
            selected.add(x)
    return selected

def _local_search(initial_set, deadline):
    """Local search: try adding missing elements and swaps to improve reciprocal sum."""
    selected = set(initial_set)
    best_sum = _reciprocal_sum(selected)
    best_set = set(selected)
    
    # Phase 1: Try adding any missing element that fits
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x not in selected:
                if not _has_3ap_with(selected, x):
                    selected.add(x)
                    changed = True
                    new_sum = _reciprocal_sum(selected)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = set(selected)
    
    # Phase 2: Try swaps - remove y, add x where x < y (so 1/x > 1/y)
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if x in selected:
            continue
        for y in sorted(selected):
            if time.time() > deadline:
                break
            if y <= x:
                continue
            # Try removing y and adding x
            new_set = selected - {y}
            if not _has_3ap_with(new_set, x):
                new_set.add(x)
                new_sum = _reciprocal_sum(new_set)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_set = set(new_set)
                    selected = set(new_set)
    
    # Phase 3: Another round of adding missing elements after swaps
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x not in selected:
                if not _has_3ap_with(selected, x):
                    selected.add(x)
                    changed = True
                    new_sum = _reciprocal_sum(selected)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = set(selected)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Start with greedy ascending
    initial = _greedy_ascending()
    
    # Apply local search
    result = _local_search(initial, deadline)
    
    return list(result)
# EVOLVE-BLOCK-END
