"""Optimized solver for large integer sets avoiding equal-lcm triples."""
import time
import random
from math import lcm
from itertools import combinations

# Precompute conflict information
N = 128

def _precompute_conflicts():
    """
    For each element x in [1, 128], compute which pairs (a, b) with a < b < x
    form a bad triple (a, b, x).
    
    A triple (a, b, x) is bad if lcm(a,b) == lcm(a,x) == lcm(b,x).
    
    We return:
    - conflict_pairs[x]: list of (a, b) such that (a, b, x) is a bad triple
    - For efficient checking during search
    """
    # conflict_with[x] = set of elements that conflict with x in some triple
    # But we need triple-level conflicts.
    
    # For each x, find all pairs (a, b) with a < b < x such that
    # lcm(a,b) == lcm(a,x) == lcm(b,x)
    conflict_pairs = [set() for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for a in range(1, x):
            for b in range(a + 1, x):
                if lcm(a, b) == lcm(a, x) == lcm(b, x):
                    conflict_pairs[x].add((a, b))
    
    return conflict_pairs

def _is_valid_selection(selected, conflict_pairs):
    """Check if a selection is valid (no bad triples)."""
    sel_set = set(selected)
    for x in selected:
        for (a, b) in conflict_pairs[x]:
            if a in sel_set and b in sel_set:
                return False
    return True

def _count_conflicts(selected, conflict_pairs):
    """Count the number of bad triples in a selection."""
    sel_set = set(selected)
    count = 0
    for x in selected:
        for (a, b) in conflict_pairs[x]:
            if a in sel_set and b in sel_set:
                count += 1
    return count

def _greedy_construct(order, conflict_pairs):
    """Greedy construction using a given order of elements."""
    selected = []
    sel_set = set()
    for x in order:
        # Check if adding x creates any bad triple
        valid = True
        for (a, b) in conflict_pairs[x]:
            if a in sel_set and b in sel_set:
                valid = False
                break
        if valid:
            selected.append(x)
            sel_set.add(x)
    return selected

def _local_search(selected, conflict_pairs, max_iters=50000, time_limit=55):
    """
    Local search: try to add elements or swap to improve.
    State: selected set.
    Move: add an unselected element if it doesn't create conflicts,
          or remove an element and try to add two others.
    """
    sel_set = set(selected)
    best_selected = list(selected)
    best_size = len(selected)
    
    start_time = time.time()
    
    # Precompute for each element, which elements it conflicts with
    # conflict_elem[x] = set of y such that there exists a bad triple involving x and y
    # Actually, we need: for element x, which pairs (a,b) in the current set
    # would form a bad triple with x.
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without creating a bad triple."""
        for (a, b) in conflict_pairs[x]:
            if a in current_set and b in current_set:
                return False
        return True
    
    def try_add_elements(current_set, current_list):
        """Try to add any unselected element."""
        for x in range(1, N + 1):
            if x not in current_set and can_add(x, current_set):
                new_set = current_set | {x}
                new_list = current_list + [x]
                return new_set, new_list, True
        return current_set, current_list, False
    
    def try_swap(current_set, current_list):
        """Try removing one element and adding one or two others."""
        for remove in list(current_set):
            new_set = current_set - {remove}
            # Try adding elements
            added = []
            for x in range(1, N + 1):
                if x not in new_set and can_add(x, new_set):
                    new_set.add(x)
                    added.append(x)
            if len(new_set) > len(current_set):
                return new_set, list(new_set), True
        return current_set, current_list, False
    
    # Simulated annealing style local search
    best_set = set(selected)
    best_list = list(selected)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try simple additions first
        new_set, new_list, improved = try_add_elements(best_set, best_list)
        if improved:
            best_set = new_set
            best_list = new_list
            best_size = len(best_list)
            continue
        
        # Try swaps: remove one, add others
        if best_list:
            remove = random.choice(best_list)
            trial_set = best_set - {remove}
            trial_list = [x for x in best_list if x != remove]
            
            # Try to add as many as possible
            for x in range(1, N + 1):
                if x not in trial_set and can_add(x, trial_set):
                    trial_set.add(x)
                    trial_list.append(x)
            
            if len(trial_list) > len(best_list):
                best_set = trial_set
                best_list = trial_list
                best_size = len(best_list)
            elif len(trial_list) == len(best_list) and random.random() < 0.1:
                # Accept same-size move with some probability for diversity
                best_set = trial_set
                best_list = trial_list
    
    return best_list

def _simulated_annealing(conflict_pairs, time_limit=50):
    """
    Simulated annealing to find large valid subsets.
    State: bitmask of selected elements.
    Energy: -size + penalty * num_conflicts
    """
    start_time = time.time()
    
    # Precompute conflict masks for fast checking
    # For each x, conflict_mask[x] is a bitmask of elements y such that
    # there's a bad triple involving x, y, and some z.
    # Actually, we need triple conflicts. Let's use a different approach.
    
    # For fast conflict checking:
    # When we consider adding x, we need to check all (a,b) in conflict_pairs[x]
    # and see if both a and b are in the current set.
    
    # Use a set for the current selection
    best_set = set()
    best_size = 0
    
    # Start with greedy
    greedy = _greedy_construct(range(1, N + 1), conflict_pairs)
    best_set = set(greedy)
    best_size = len(greedy)
    
    # SA
    current_set = set(greedy)
    current_size = len(greedy)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    def count_conflicts_in_set(s):
        """Count bad triples in set s."""
        c = 0
        for x in s:
            for (a, b) in conflict_pairs[x]:
                if a in s and b in s:
                    c += 1
        return c
    
    # Precompute for faster conflict count
    # conflict_count[x][y] = 1 if x and y are in a bad triple together (with some z)
    # Actually, let's precompute: for each pair (x, y), the number of z such that
    # (x, y, z) is a bad triple.
    pair_conflict_count = {}
    for z in range(1, N + 1):
        for (a, b) in conflict_pairs[z]:
            key = (min(a, b), max(a, b))
            pair_conflict_count[key] = pair_conflict_count.get(key, 0) + 1
    
    def num_conflicts(s):
        """Count bad triples in set s using pair conflicts."""
        c = 0
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a, b = s_list[i], s_list[j]
                key = (min(a, b), max(a, b))
                if key in pair_conflict_count:
                    # For each z that forms a bad triple with a and b,
                    # check if z is also in s
                    # This is O(|s|) per pair, which is slow.
                    # Let's use the original method.
                    pass
        # Use original method
        c = 0
        for x in s:
            for (a, b) in conflict_pairs[x]:
                if a in s and b in s:
                    c += 1
        return c
    
    # Better: precompute for each element x, the set of elements that can form
    # a bad triple with x and some third element.
    # triple_partners[x] = set of y such that exists z with (x,y,z) bad triple
    # And for counting, we need to know for each pair (x,y) how many z complete it.
    
    # Let's just use the direct method but optimize with bitmasks.
    # Represent set as a 128-bit integer.
    
    # conflict_mask[x]: for each x, a bitmask where bit y is set if there exists
    # a bad triple (x, y, z) for some z. But we need to know if BOTH partners are present.
    
    # Alternative: for each x, store list of (a, b) pairs. When checking if x can
    # be added to set S, check if any (a,b) in conflict_pairs[x] has both a,b in S.
    
    # For SA, let's use a simpler approach: try random moves.
    
    for step in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        T = max(T * cooling, T_min)
        
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = current_set | {x}
            delta_conf = num_conflicts(new_set) - num_conflicts(current_set)
            delta_size = 1
            # Energy change: -delta_size + penalty * delta_conf
            penalty = 10
            delta_energy = -delta_size + penalty * delta_conf
            if delta_energy <= 0 or random.random() < min(1.0, max(0.0, -delta_energy / T)):
                current_set = new_set
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
        elif move_type < 0.7:
            # Remove a random element
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            delta_conf = num_conflicts(new_set) - num_conflicts(current_set)
            delta_size = -1
            penalty = 10
            delta_energy = -delta_size + penalty * delta_conf
            if delta_energy <= 0 or random.random() < min(1.0, max(0.0, -delta_energy / T)):
                current_set = new_set
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
        else:
            # Swap: remove one, add one
            if not current_set:
                continue
            remove = random.choice(list(current_set))
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            add = random.choice(candidates)
            new_set = (current_set - {remove}) | {add}
            delta_conf = num_conflicts(new_set) - num_conflicts(current_set)
            delta_size = 0
            penalty = 10
            delta_energy = -delta_size + penalty * delta_conf
            if delta_energy <= 0 or random.random() < min(1.0, max(0.0, -delta_energy / T)):
                current_set = new_set
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
    
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute conflicts
    conflict_pairs = _precompute_conflicts()
    
    # Try multiple greedy orders
    best_result = []
    
    # Greedy with natural order
    r1 = _greedy_construct(range(1, N + 1), conflict_pairs)
    if len(r1) > len(best_result):
        best_result = r1
    
    # Greedy with reverse order
    r2 = _greedy_construct(range(N, 0, -1), conflict_pairs)
    if len(r2) > len(best_result):
        best_result = r2
    
    # Greedy with random orders
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_construct(order, conflict_pairs)
        if len(r) > len(best_result):
            best_result = r
    
    # Local search from best greedy
    if time.time() < deadline - 5:
        ls_result = _local_search(best_result, conflict_pairs, 
                                   max_iters=100000, 
                                   time_limit=min(20, deadline - time.time() - 2))
        if len(ls_result) > len(best_result):
            best_result = ls_result
    
    # Simulated annealing
    if time.time() < deadline - 3:
        sa_result = _simulated_annealing(conflict_pairs, 
                                          time_limit=min(30, deadline - time.time() - 2))
        if len(sa_result) > len(best_result):
            best_result = sa_result
    
    # Final local search from best
    if time.time() < deadline - 1 and best_result:
        final_ls = _local_search(best_result, conflict_pairs,
                                  max_iters=50000,
                                  time_limit=min(10, deadline - time.time() - 0.5))
        if len(final_ls) > len(best_result):
            best_result = final_ls
    
    # Ensure nonempty
    if not best_result:
        best_result = [1]
    
    return best_result
