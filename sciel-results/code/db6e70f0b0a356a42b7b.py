"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _conflict_count(selected_set, x):
    """Count how many conflicts x has with the current selected set.
    A conflict exists if:
    - x divides sum of any 3 distinct other selected elements, OR
    - any selected element divides sum of x + any 2 other selected elements
    """
    n_conflicts = 0
    elems = [e for e in selected_set if e != x]
    # Check: x divides sum of any 3 from elems
    for tri in combinations(elems, 3):
        if sum(tri) % x == 0:
            n_conflicts += 1
    # Check: any elem divides sum of (x + 2 other elems)
    for a in elems:
        rest = [e for e in elems if e != a]
        for pair in combinations(rest, 2):
            if (x + pair[0] + pair[1]) % a == 0:
                n_conflicts += 1
    return n_conflicts

def _is_valid(selected):
    """Check if the full set is valid."""
    for a in selected:
        others = [e for e in selected if e != a]
        for tri in combinations(others, 3):
            if sum(tri) % a == 0:
                return False
    return True

def _greedy_init():
    """Greedy from top: add numbers from 192 down to 1 if they don't create conflicts."""
    selected = []
    selected_set = set()
    for x in range(192, 0, -1):
        if _conflict_count(selected_set, x) == 0:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, time_limit):
    """Local search: try swaps and additions to improve cardinality."""
    selected_set = set(selected)
    best = list(selected)
    best_set = set(best)
    best_size = len(best)
    start_time = time.time()
    
    # Try adding elements
    for x in range(1, 193):
        if time.time() - start_time > time_limit:
            break
        if x in selected_set:
            continue
        if _conflict_count(selected_set, x) == 0:
            best.append(x)
            best_set.add(x)
            best_size += 1
    
    # Try swaps: remove one, try to add two
    for i in range(len(best)):
        if time.time() - start_time > time_limit:
            break
        a = best[i]
        trial_set = best_set - {a}
        # Try to add two elements
        for x in range(1, 193):
            if time.time() - start_time > time_limit:
                break
            if x in trial_set or x == a:
                continue
            if _conflict_count(trial_set, x) == 0:
                trial_set2 = trial_set | {x}
                for y in range(1, 193):
                    if time.time() - start_time > time_limit:
                        break
                    if y in trial_set2 or y == a or y == x:
                        continue
                    if _conflict_count(trial_set2, y) == 0:
                        new_set = trial_set2 | {y}
                        if len(new_set) > best_size:
                            best = list(new_set)
                            best_set = new_set
                            best_size = len(best)
                        break
        # Try to add one element (net zero, but might enable future improvements)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    # Phase 1: Greedy initialization
    selected = _greedy_init()
    best = list(selected)
    best_size = len(best)
    
    # Phase 2: Local search improvements
    selected = _local_search(selected, 30)
    if len(selected) > best_size:
        best = list(selected)
        best_size = len(best)
    
    # Phase 3: Multiple random restarts with perturbation
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Perturb the best solution: remove 1-3 random elements
        perturbed = list(best)
        k_remove = random.randint(1, 3)
        for _ in range(k_remove):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Run local search on perturbed
        result = _local_search(perturbed, 10)
        if len(result) > best_size:
            best = list(result)
            best_size = len(best)
    
    # Phase 4: Final greedy fill
    best_set = set(best)
    for x in range(1, 193):
        if time.time() > deadline - 2:
            break
        if x not in best_set and _conflict_count(best_set, x) == 0:
            best.append(x)
            best_set.add(x)
    
    return best
# EVOLVE-BLOCK-END
