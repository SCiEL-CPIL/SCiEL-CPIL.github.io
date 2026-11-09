"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _count_3ap(sel):
    """Count distinct 3-term APs with positive difference in set sel."""
    count = 0
    s = set(sel)
    for a in s:
        for c in s:
            if c <= a:
                continue
            d = c - a
            if d % 2 != 0:
                continue
            mid = a + d // 2
            if mid in s and mid != a and mid != c:
                count += 1
    return count

def _valid_add(sel, x):
    """Check if adding x to sel creates no 4-term AP."""
    s = sel
    for d in range(1, 257):
        if x + d in s and x + 2*d in s and x + 3*d in s:
            return False
        if x - d >= 1 and x - d in s and x + d in s and x + 2*d in s:
            return False
        if x - 2*d >= 1 and x - 2*d in s and x - d in s and x + d in s:
            return False
        if x - 3*d >= 1 and x - 3*d in s and x - 2*d in s and x - d in s:
            return False
    return True

def _valid_swap(sel, x_remove, x_add):
    """Check if removing x_remove and adding x_add keeps no 4-term AP."""
    new_sel = set(sel)
    new_sel.discard(x_remove)
    # Check 4-APs involving x_add
    for d in range(1, 257):
        if x_add + d in new_sel and x_add + 2*d in new_sel and x_add + 3*d in new_sel:
            return False
        if x_add - d >= 1 and x_add - d in new_sel and x_add + d in new_sel and x_add + 2*d in new_sel:
            return False
        if x_add - 2*d >= 1 and x_add - 2*d in new_sel and x_add - d in new_sel and x_add + d in new_sel:
            return False
        if x_add - 3*d >= 1 and x_add - 3*d in new_sel and x_add - 2*d in new_sel and x_add - d in new_sel:
            return False
    # Check 4-APs that were broken by removing x_remove but might now exist with x_add
    # Actually, we need to check all 4-APs in new_sel, but that's expensive.
    # Instead, check if any 4-AP exists that doesn't involve x_add (these were already valid)
    # and any 4-AP involving x_add (checked above). So this should be sufficient.
    return True

def _greedy_init():
    selected = set()
    for x in range(1, 257):
        if _valid_add(selected, x):
            selected.add(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_set = _greedy_init()
    best_score = _count_3ap(best_set)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    current_set = set(best_set)
    current_score = best_score
    temperature = 100.0
    cooling = 0.99
    rng = random.Random(42)
    
    while time.time() < deadline - 2:
        move_type = rng.random()
        
        if move_type < 0.35:
            # Add
            candidates = [x for x in range(1, 257) if x not in current_set and _valid_add(current_set, x)]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_score = _count_3ap(current_set | {x})
            delta = new_score - current_score
            if delta >= 0 or rng.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                current_set.add(x)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
        elif move_type < 0.55:
            # Remove
            if not current_set:
                continue
            x = rng.choice(list(current_set))
            new_score = _count_3ap(current_set - {x})
            delta = new_score - current_score
            if delta >= 0 or rng.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                current_set.discard(x)
                current_score = new_score
        else:
            # Swap
            if not current_set:
                continue
            x = rng.choice(list(current_set))
            candidates = [y for y in range(1, 257) if y not in current_set and y != x]
            if not candidates:
                continue
            y = rng.choice(candidates)
            if _valid_swap(current_set, x, y):
                new_score = _count_3ap((current_set - {x}) | {y})
                delta = new_score - current_score
                if delta >= 0 or rng.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                    current_set.discard(x)
                    current_set.add(y)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
        
        temperature *= cooling
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
