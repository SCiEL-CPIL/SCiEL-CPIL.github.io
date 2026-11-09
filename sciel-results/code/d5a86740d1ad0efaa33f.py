import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """
    Precompute all conflict triples (a, b, c) where a, b, c are distinct,
    1 <= a,b,c <= 180, and 1/a = 1/b + 1/c.
    Returns a list of (a, b, c) tuples.
    """
    conflicts = []
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, 181):
            if a2 % d != 0:
                continue
            c_val = a + a2 // d
            b_val = a + d
            if b_val > 180 or c_val > 180:
                continue
            if b_val <= 0 or c_val <= 0:
                continue
            if b_val == a or c_val == a:
                continue
            if b_val == c_val:
                continue
            conflicts.append((a, b_val, c_val))
    return conflicts

def _build_conflict_graph(conflicts):
    """
    Build data structures for fast conflict checking.
    """
    conflict_triples = []
    elem_to_conflicts = {i: [] for i in range(1, 181)}
    
    for (a, b, c) in conflicts:
        triple = frozenset((a, b, c))
        idx = len(conflict_triples)
        conflict_triples.append(triple)
        elem_to_conflicts[a].append(idx)
        elem_to_conflicts[b].append(idx)
        elem_to_conflicts[c].append(idx)
    
    return conflict_triples, elem_to_conflicts

def _incremental_check_valid(s_set, new_elem, conflict_triples, elem_to_conflicts):
    """
    Check if adding new_elem to s_set creates any conflict.
    Only need to check conflicts that involve new_elem.
    """
    for idx in elem_to_conflicts[new_elem]:
        triple = conflict_triples[idx]
        others = triple - {new_elem}
        if others.issubset(s_set):
            return False
    return True

def _check_valid(s_set, conflict_triples):
    """
    Check if s_set has no conflict triple fully contained.
    """
    for triple in conflict_triples:
        if triple.issubset(s_set):
            return False
    return True

def _greedy_build(elements, conflict_triples, elem_to_conflicts, deadline):
    """
    Greedily build a valid set by adding elements in the given order.
    """
    current = set()
    for x in elements:
        if time.time() > deadline:
            break
        if x in current:
            continue
        if _incremental_check_valid(current, x, conflict_triples, elem_to_conflicts):
            current.add(x)
    return current

def _local_search_1swap(current_set, conflict_triples, elem_to_conflicts, all_elements, deadline, max_rounds=100):
    """
    Local search: try to add missing elements, and if stuck, try 1-swap moves.
    """
    current = set(current_set)
    best = set(current)
    all_elems_set = set(all_elements)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Try to add any missing element
        missing = [x for x in all_elements if x not in current]
        random.shuffle(missing)
        added_any = False
        for m in missing:
            if time.time() > deadline:
                break
            if _incremental_check_valid(current, m, conflict_triples, elem_to_conflicts):
                current.add(m)
                added_any = True
                if len(current) > len(best):
                    best = set(current)
        
        if added_any:
            continue
        
        # Stuck: try 1-swap moves (remove x, add y)
        in_set = list(current)
        random.shuffle(in_set)
        found_improvement = False
        
        for x in in_set:
            if time.time() > deadline:
                break
            temp = current - {x}
            missing_after = [y for y in all_elements if y not in temp]
            random.shuffle(missing_after)
            
            # Try to add as many as possible after removing x
            temp_set = set(temp)
            for y in missing_after:
                if time.time() > deadline:
                    break
                if _incremental_check_valid(temp_set, y, conflict_triples, elem_to_conflicts):
                    temp_set.add(y)
            
            if len(temp_set) > len(current):
                current = temp_set
                found_improvement = True
                if len(current) > len(best):
                    best = set(current)
                break
        
        if not found_improvement:
            break
    
    return best

def _local_search_2swap(current_set, conflict_triples, elem_to_conflicts, all_elements, deadline, max_rounds=50):
    """
    Local search with 2-swap moves: remove two elements, add as many as possible.
    """
    current = set(current_set)
    best = set(current)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Try to add any missing element first
        missing = [x for x in all_elements if x not in current]
        random.shuffle(missing)
        added_any = False
        for m in missing:
            if time.time() > deadline:
                break
            if _incremental_check_valid(current, m, conflict_triples, elem_to_conflicts):
                current.add(m)
                added_any = True
                if len(current) > len(best):
                    best = set(current)
        
        if added_any:
            continue
        
        # Try 2-swap: remove x1, x2 from current, then greedily add
        in_set = list(current)
        random.shuffle(in_set)
        found_improvement = False
        
        for i in range(min(len(in_set), 30)):
            if time.time() > deadline:
                break
            x1 = in_set[i]
            for j in range(i+1, min(len(in_set), 30)):
                if time.time() > deadline:
                    break
                x2 = in_set[j]
                temp = current - {x1, x2}
                missing_after = [y for y in all_elements if y not in temp]
                random.shuffle(missing_after)
                
                temp_set = set(temp)
                for y in missing_after:
                    if time.time() > deadline:
                        break
                    if _incremental_check_valid(temp_set, y, conflict_triples, elem_to_conflicts):
                        temp_set.add(y)
                
                if len(temp_set) > len(current):
                    current = temp_set
                    found_improvement = True
                    if len(current) > len(best):
                        best = set(current)
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return best

def _simulated_annealing(current_set, conflict_triples, elem_to_conflicts, all_elements, deadline, 
                          initial_temp=1.0, cooling=0.95, max_iters=5000):
    """
    Simulated annealing to escape local optima.
    Moves: add one element, remove one element, or swap one element.
    Objective: maximize set size.
    """
    current = set(current_set)
    best = set(current)
    best_size = len(current)
    temp = initial_temp
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random missing element
            missing = [x for x in all_elements if x not in current]
            if not missing:
                continue
            x = random.choice(missing)
            if _incremental_check_valid(current, x, conflict_triples, elem_to_conflicts):
                new_set = current | {x}
                delta = 1  # improvement
                current = new_set
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
            # If invalid, skip (no change)
        elif move_type < 0.8:
            # Try to remove a random element and add a different one
            if not current:
                continue
            x = random.choice(list(current))
            temp_set = current - {x}
            missing = [y for y in all_elements if y not in temp_set]
            if not missing:
                continue
            y = random.choice(missing)
            if _incremental_check_valid(temp_set, y, conflict_triples, elem_to_conflicts):
                new_set = temp_set | {y}
                delta = 0  # neutral swap
                # Accept with some probability to explore
                if random.random() < math.exp(0 / temp if temp > 0 else 1):
                    current = new_set
            # If invalid, skip
        else:
            # Try to remove a random element (to explore smaller sets that may lead to larger ones)
            if not current:
                continue
            x = random.choice(list(current))
            new_set = current - {x}
            delta = -1  # worsening
            accept_prob = math.exp(delta / temp) if temp > 0 else 0
            if random.random() < accept_prob:
                current = new_set
        
        temp *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s margin before 90s kill
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    conflict_triples, elem_to_conflicts = _build_conflict_graph(conflicts)
    
    all_elements = list(range(1, 181))
    
    best_set = set()
    best_size = 0
    
    # Structural initialization: elements >= 90 can't be the min in any conflict
    # Start with all elements >= 90, then greedily add smaller ones
    structural_init = set(range(90, 181))
    # Greedily add smaller elements
    for x in range(1, 90):
        if time.time() > deadline:
            break
        if _incremental_check_valid(structural_init, x, conflict_triples, elem_to_conflicts):
            structural_init.add(x)
    
    if len(structural_init) > best_size:
        best_size = len(structural_init)
        best_set = set(structural_init)
    
    # Multiple greedy orders
    # Order 1: descending
    current = _greedy_build(all_elements[::-1], conflict_triples, elem_to_conflicts, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Order 2: ascending
    current = _greedy_build(all_elements, conflict_triples, elem_to_conflicts, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Order 3: random restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        random.seed(1000 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        current = _greedy_build(shuffled, conflict_triples, elem_to_conflicts, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Local search 1-swap from best
    improved = _local_search_1swap(best_set, conflict_triples, elem_to_conflicts, all_elements, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    # Local search 2-swap from best
    improved = _local_search_2swap(best_set, conflict_triples, elem_to_conflicts, all_elements, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    # Simulated annealing from best
    improved = _simulated_annealing(best_set, conflict_triples, elem_to_conflicts, all_elements, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    # More local search restarts from different greedy solutions
    for restart in range(15):
        if time.time() > deadline:
            break
        random.seed(2000 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        current = _greedy_build(shuffled, conflict_triples, elem_to_conflicts, deadline)
        improved = _local_search_1swap(current, conflict_triples, elem_to_conflicts, all_elements, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        improved = _local_search_2swap(best_set, conflict_triples, elem_to_conflicts, all_elements, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Final SA polish from best
    if time.time() < deadline - 2:
        improved = _simulated_annealing(best_set, conflict_triples, elem_to_conflicts, all_elements, deadline,
                                         initial_temp=0.5, cooling=0.98, max_iters=3000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    return list(best_set)

# EVOLVE-BLOCK-END
