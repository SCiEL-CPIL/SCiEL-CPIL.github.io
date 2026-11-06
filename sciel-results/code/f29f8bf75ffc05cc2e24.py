"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """For each element x, find all pairs (y,z) such that {x,y,z} is a bad triple.
    conflicts[x] = set of frozenset({y,z}) for each bad triple containing x."""
    conflicts = {x: set() for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for pair in conflicts[x]:
        if pair & selected_set == pair:
            return False
    return True

def _greedy(conflicts, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = []
    selected_set = set()
    for x in order:
        if _can_add(x, selected_set, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, conflicts, time_limit=10):
    """Local search: try removing one element and adding another to improve size."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try all single removals
        for rem in list(selected_set):
            candidate = selected_set - {rem}
            # Try to add any element not in candidate
            for x in range(1, 129):
                if x in candidate:
                    continue
                if _can_add(x, candidate, conflicts):
                    new_set = candidate | {x}
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try double removals
            improved = False
            for rem1, rem2 in combinations(list(selected_set), 2):
                candidate = selected_set - {rem1, rem2}
                for x in range(1, 129):
                    if x in candidate:
                        continue
                    if _can_add(x, candidate, conflicts):
                        new_set = candidate | {x}
                        if len(new_set) > best_size:
                            best = list(new_set)
                            best_size = len(best)
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    return best

def _anneal(selected, conflicts, time_limit=15):
    """Simulated annealing with better move generation."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    start = time.time()
    current_temp = 5.0
    cooling = 0.97
    
    while time.time() - start < time_limit and current_temp > 0.1:
        current_temp *= cooling
        
        if not selected_set:
            for x in range(1, 129):
                if _can_add(x, selected_set, conflicts):
                    selected_set.add(x)
                    break
            continue
        
        # Random move: remove 1-2 random elements, try to add 1-2 others
        n_remove = random.randint(1, 2)
        to_remove = random.sample(list(selected_set), min(n_remove, len(selected_set)))
        candidate = selected_set - set(to_remove)
        
        # Try to add random elements
        available = [x for x in range(1, 129) if x not in candidate]
        n_add = random.randint(1, 2)
        to_add = random.sample(available, min(n_add, len(available)))
        
        new_set = set(candidate)
        for x in to_add:
            if _can_add(x, new_set, conflicts):
                new_set.add(x)
        
        # Try to greedily add more
        for y in range(1, 129):
            if y not in new_set and _can_add(y, new_set, conflicts):
                new_set.add(y)
        
        if len(new_set) > best_size:
            best = list(new_set)
            best_size = len(best)
            selected_set = new_set
        elif random.random() < current_temp / 10.0:
            selected_set = new_set
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute conflict structure
    conflicts = _build_conflicts()
    
    best = []
    best_size = 0
    
    # Strategy 1: Greedy with natural order
    sel = _greedy(conflicts, list(range(1, 129)))
    best = sel
    best_size = len(sel)
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline - 30:
        sel = _greedy(conflicts, list(range(128, 0, -1)))
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Strategy 3: Local search on best
    if time.time() < deadline - 25:
        improved = _local_search(best, conflicts, time_limit=8)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Strategy 4: Multiple random restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 20 and restarts < 200:
        restarts += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Improve if close to best
        if len(sel) >= best_size - 3 and time.time() < deadline - 25:
            improved = _local_search(sel, conflicts, time_limit=3)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Strategy 5: Simulated annealing on best solution
    if time.time() < deadline - 10 and len(best) > 0:
        improved = _anneal(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Strategy 6: Final local search
    if time.time() < deadline - 3 and len(best) > 0:
        improved = _local_search(best, conflicts, time_limit=3)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    return best
# EVOLVE-BLOCK-END
