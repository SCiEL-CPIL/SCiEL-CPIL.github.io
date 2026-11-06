"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """
    For each element x, precompute the set of pairs (y,z) such that {x,y,z} is a bad triple.
    conflicts[x] = set of (y, z) tuples where y < z, and {x,y,z} forms a bad triple.
    """
    conflicts = {x: set() for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            # For each element in the triple, store the other two
            conflicts[a].add((b, c))
            conflicts[b].add((a, c))
            conflicts[c].add((a, b))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for (y, z) in conflicts[x]:
        if y in selected_set and z in selected_set:
            return False
    return True

def _greedy(conflicts, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = set()
    for x in order:
        if _can_add(x, selected, conflicts):
            selected.add(x)
    return selected

def _local_search(selected, conflicts, time_limit=15):
    """
    Local search: try swaps (remove one, add one) and additions.
    Returns improved set or the original if no improvement.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try adding any element
        for x in range(1, 129):
            if x in selected:
                continue
            if _can_add(x, selected, conflicts):
                selected.add(x)
                if len(selected) > best_size:
                    best = set(selected)
                    best_size = len(selected)
                improved = True
        
        # Try swaps: remove one, add one
        if not improved:
            for rem in list(selected):
                candidate = selected - {rem}
                for x in range(1, 129):
                    if x in candidate:
                        continue
                    if _can_add(x, candidate, conflicts):
                        new_set = candidate | {x}
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(best)
                            selected = new_set
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def _simulated_annealing(conflicts, time_limit=20):
    """
    Simulated annealing: start from a greedy solution, then do random swaps
    accepting worse solutions with some probability to escape local optima.
    """
    # Start from a greedy solution
    order = list(range(1, 129))
    random.shuffle(order)
    current = _greedy(conflicts, order)
    best = set(current)
    best_size = len(current)
    
    start = time.time()
    temp = 10.0
    cooling = 0.99
    
    while time.time() - start < time_limit:
        temp *= cooling
        if temp < 0.01:
            break
        
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random element
            candidates = [x for x in range(1, 129) if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            if _can_add(x, current, conflicts):
                current.add(x)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            else:
                # Can't add, try swap
                if current:
                    rem = random.choice(list(current))
                    candidate = current - {rem}
                    if _can_add(x, candidate, conflicts):
                        current = candidate | {x}
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
        
        elif move_type < 0.6:
            # Remove a random element
            if current:
                rem = random.choice(list(current))
                current.discard(rem)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
        
        else:
            # Swap: remove one random, try to add another
            if current:
                rem = random.choice(list(current))
                candidate = current - {rem}
                candidates = [x for x in range(1, 129) if x not in candidate]
                if candidates:
                    x = random.choice(candidates)
                    if _can_add(x, candidate, conflicts):
                        current = candidate | {x}
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
                    else:
                        # Accept with probability based on temperature (allow removal)
                        if random.random() < temp / 10.0:
                            current = candidate
    
    return best

def _is_valid(selected, conflicts):
    """Check if selected set has no bad triple."""
    sel_set = set(selected)
    for x in sel_set:
        for (y, z) in conflicts[x]:
            if y in sel_set and z in sel_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    conflicts = _build_conflicts()
    
    best = set()
    best_size = 0
    
    # Multiple greedy starts with different orderings
    random.seed(42)
    
    # First, try natural order
    sel = _greedy(conflicts, list(range(1, 129)))
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # Try reverse order
    if time.time() < deadline - 5:
        sel = _greedy(conflicts, list(range(128, 0, -1)))
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Multiple random greedy starts with local search
    for restart in range(200):
        if time.time() > deadline - 3:
            break
        
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Local search on promising solutions
        if len(sel) >= best_size - 2 and time.time() < deadline - 8:
            improved = _local_search(sel, conflicts, time_limit=5)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Simulated annealing on the best solution found
    if time.time() < deadline - 10:
        sa_result = _simulated_annealing(conflicts, time_limit=20)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Final local search on the best
    if time.time() < deadline - 5 and best:
        improved = _local_search(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    # Validate
    if not _is_valid(best, conflicts):
        # Fallback to a simple greedy
        best = _greedy(conflicts, list(range(1, 129)))
    
    return list(best)
# EVOLVE-BLOCK-END
