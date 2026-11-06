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
    conflicts[x] = list of (y, z) pairs.
    Also return the total number of bad triples for reference.
    """
    conflicts = {x: [] for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            conflicts[a].append((b, c))
            conflicts[b].append((a, c))
            conflicts[c].append((a, b))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for (y, z) in conflicts[x]:
        if y in selected_set and z in selected_set:
            return False
    return True

def _greedy_initial(conflicts, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = []
    selected_set = set()
    for x in order:
        if _can_add(x, selected_set, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_improve(selected, conflicts, time_limit=30):
    """Try to improve by removing one element and adding one or more others."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try swap: remove one, add one
        for rem in list(selected_set):
            candidate_set = selected_set - {rem}
            for x in range(1, 129):
                if x in candidate_set:
                    continue
                if _can_add(x, candidate_set, conflicts):
                    new_set = candidate_set | {x}
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try double-swap: remove two, add two
            improved2 = False
            elems = list(selected_set)
            for i in range(len(elems)):
                if time.time() - start > time_limit:
                    break
                for j in range(i+1, len(elems)):
                    rem1, rem2 = elems[i], elems[j]
                    candidate_set = selected_set - {rem1, rem2}
                    for x in range(1, 129):
                        if x in candidate_set:
                            continue
                        if not _can_add(x, candidate_set, conflicts):
                            continue
                        temp_set = candidate_set | {x}
                        for y in range(1, 129):
                            if y in temp_set:
                                continue
                            if _can_add(y, temp_set, conflicts):
                                new_set = temp_set | {y}
                                if len(new_set) > best_size:
                                    best = list(new_set)
                                    best_size = len(best)
                                    selected_set = new_set
                                    improved2 = True
                                    break
                        if improved2:
                            break
                    if improved2:
                        break
                if improved2:
                    break
            if not improved2:
                break
    
    return best

def _simulated_annealing(selected, conflicts, time_limit=20):
    """Simulated annealing to escape local optima."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    
    start = time.time()
    temp = 5.0
    cooling = 0.999
    
    while time.time() - start < time_limit:
        temp *= cooling
        if temp < 0.1:
            break
        
        # Random move: remove one element
        if selected_set:
            rem = random.choice(list(selected_set))
            new_set = selected_set - {rem}
            # Try to add one or more elements
            for _ in range(random.randint(1, 3)):
                candidates = [x for x in range(1, 129) if x not in new_set and _can_add(x, new_set, conflicts)]
                if not candidates:
                    break
                x = random.choice(candidates)
                new_set.add(x)
            
            # Accept if better or with probability
            if len(new_set) > len(selected_set):
                selected_set = new_set
                if len(selected_set) > best_size:
                    best = list(selected_set)
                    best_size = len(best)
            else:
                # Accept with probability exp(-delta/temp)
                delta = len(selected_set) - len(new_set)
                if random.random() < (2.718281828 ** (-delta / temp)):
                    selected_set = new_set
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    conflicts = _build_conflicts()
    
    best = []
    best_size = 0
    
    # Initial greedy with natural order
    sel = _greedy_initial(conflicts, list(range(1, 129)))
    best = sel
    best_size = len(sel)
    
    # Local improvement
    if time.time() < deadline - 15:
        improved = _local_improve(best, conflicts, time_limit=15)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Multiple restarts with different orderings
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy_initial(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Local improve if promising
        if len(sel) >= best_size - 2 and time.time() < deadline - 20:
            improved = _local_improve(sel, conflicts, time_limit=5)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Final local improvement on best solution
    if time.time() < deadline - 5 and len(best) > 0:
        improved = _local_improve(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Simulated annealing if time permits
    if time.time() < deadline - 5 and len(best) > 0:
        improved = _simulated_annealing(best, conflicts, time_limit=10)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    # Final validation
    bad_triples = set()
    for x in range(1, 129):
        for (y, z) in conflicts[x]:
            triple = tuple(sorted([x, y, z]))
            bad_triples.add(triple)
    
    best_set = set(best)
    valid = True
    for (a, b, c) in bad_triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
    
    if not valid:
        best = _greedy_initial(conflicts, list(range(1, 129)))
    
    return best
# EVOLVE-BLOCK-END
