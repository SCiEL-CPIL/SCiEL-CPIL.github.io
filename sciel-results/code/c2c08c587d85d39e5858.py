"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    """
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_triple_index(triples):
    """
    Build index: element -> list of triple indices containing it.
    Also returns the list of triples.
    """
    n_triples = len(triples)
    elem_to_triples = {}
    for i, t in enumerate(triples):
        for x in t:
            if x not in elem_to_triples:
                elem_to_triples[x] = []
            elem_to_triples[x].append(i)
    return triples, elem_to_triples

def _is_valid_selection(selected_set, triples, elem_to_triples):
    """
    Check if selected_set is valid (no bad triple fully contained in it).
    Returns True if valid.
    """
    # A set is invalid if there exists a triple (a,b,c) such that a,b,c are all in selected_set.
    # Equivalently, the removed set must hit every bad triple.
    # So we check: for every triple, at least one element is NOT in selected_set.
    for t in triples:
        if t[0] in selected_set and t[1] in selected_set and t[2] in selected_set:
            return False
    return True

def _count_conflicts(selected_set, triples, elem_to_triples):
    """
    Count number of bad triples fully contained in selected_set.
    """
    count = 0
    for t in triples:
        if t[0] in selected_set and t[1] in selected_set and t[2] in selected_set:
            count += 1
    return count

def _greedy_hit_set(triples, elem_to_triples, seed=0):
    """
    Greedy hitting set: repeatedly remove the element that hits the most remaining bad triples.
    """
    rng = random.Random(seed)
    active_triples = set(range(len(triples)))
    removed = set()
    
    # Precompute for each triple its elements
    # We need to efficiently find which element hits the most active triples.
    
    while active_triples:
        freq = {}
        for ti in active_triples:
            t = triples[ti]
            for x in t:
                freq[x] = freq.get(x, 0) + 1
        
        max_freq = max(freq.values())
        candidates = [x for x, f in freq.items() if f == max_freq]
        x = rng.choice(candidates)
        
        removed.add(x)
        
        # Remove all active triples containing x
        new_active = set()
        for ti in active_triples:
            t = triples[ti]
            if x not in t:
                new_active.add(ti)
        active_triples = new_active
        
    return removed

def _simulated_annealing(initial_removed, triples, elem_to_triples, max_n, rng, time_limit):
    """
    Simulated annealing to minimize |removed| (maximize |selected|).
    State: removed set.
    Moves: 
      1. Remove an element from removed (add to selected) - may create conflicts.
      2. Add an element to removed (remove from selected) - always reduces conflicts.
      3. Swap: remove x from removed, add y to removed.
    Objective: minimize |removed|, subject to validity (0 conflicts).
    We use a penalty for conflicts: cost = |removed| + penalty * conflicts.
    """
    start = time.time()
    current_removed = set(initial_removed)
    current_conflicts = _count_conflicts(set(range(1, max_n+1)) - current_removed, triples, elem_to_triples)
    current_cost = len(current_removed) + 100 * current_conflicts
    
    best_removed = set(current_removed)
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    if current_conflicts == 0:
        best_removed = set(current_removed)
        best_cost = len(current_removed)
    
    T_start = 5.0
    T_end = 0.01
    cooling = 0.995
    
    T = T_start
    step = 0
    max_steps = 50000  # Hard cap to avoid infinite loop if time check fails
    
    while T > T_end and (time.time() - start) < time_limit and step < max_steps:
        step += 1
        
        # Choose a random move
        all_elements = list(range(1, max_n + 1))
        move_type = rng.random()
        
        if move_type < 0.4 and current_removed:
            # Remove from removed (add to selected)
            candidates = list(current_removed)
            x = rng.choice(candidates)
            new_removed = set(current_removed)
            new_removed.discard(x)
        elif move_type < 0.7 and len(current_removed) < max_n:
            # Add to removed (remove from selected)
            candidates = [x for x in all_elements if x not in current_removed]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_removed = set(current_removed)
            new_removed.add(x)
        else:
            # Swap
            if current_removed and len(current_removed) < max_n:
                candidates_remove = list(current_removed)
                candidates_add = [x for x in all_elements if x not in current_removed]
                if not candidates_add:
                    continue
                x_out = rng.choice(candidates_remove)
                x_in = rng.choice(candidates_add)
                new_removed = set(current_removed)
                new_removed.discard(x_out)
                new_removed.add(x_in)
            else:
                # Fallback to simple add/remove
                if current_removed:
                    x = rng.choice(list(current_removed))
                    new_removed = set(current_removed)
                    new_removed.discard(x)
                else:
                    x = rng.choice(all_elements)
                    new_removed = set(current_removed)
                    new_removed.add(x)
        
        new_selected = set(range(1, max_n + 1)) - new_removed
        new_conflicts = _count_conflicts(new_selected, triples, elem_to_triples)
        new_cost = len(new_removed) + 100 * new_conflicts
        
        delta_cost = new_cost - current_cost
        
        if delta_cost <= 0 or rng.random() < math.exp(-delta_cost / T):
            current_removed = new_removed
            current_conflicts = new_conflicts
            current_cost = new_cost
            
            # Update best
            if new_conflicts == 0 and len(new_removed) < len(best_removed):
                best_removed = set(new_removed)
                best_cost = len(new_removed)
                best_conflicts = 0
        
        T *= cooling
        
    # If best has conflicts, we need to fix it
    if best_conflicts > 0:
        # Greedily fix conflicts in best_removed
        # This might increase size, but ensures validity
        fixed_removed = set(best_removed)
        fixed_selected = set(range(1, max_n + 1)) - fixed_removed
        active_conflicts = []
        for t in triples:
            if t[0] in fixed_selected and t[1] in fixed_selected and t[2] in fixed_selected:
                active_conflicts.append(t)
        
        # Greedy hitting set on active_conflicts
        while active_conflicts:
            freq = {}
            for t in active_conflicts:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            max_f = max(freq.values())
            candidates = [x for x, f in freq.items() if f == max_f]
            x = rng.choice(candidates)
            fixed_removed.add(x)
            fixed_selected.discard(x)
            new_active = []
            for t in active_conflicts:
                if x not in t:
                    new_active.append(t)
            active_conflicts = new_active
        
        best_removed = fixed_removed
    
    return best_removed

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe (90s kill, leave margin)
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    triples, elem_to_triples = _build_triple_index(bad_triples)
    
    best_removed = None
    best_size = -1
    
    # Phase 1: Greedy hitting set with multiple seeds
    num_greedy = 50
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        removed = _greedy_hit_set(triples, elem_to_triples, seed=i)
        size = max_n - len(removed)
        if size > best_size:
            best_size = size
            best_removed = removed
    
    if best_removed is None:
        # Fallback
        selected = []
        for x in range(1, max_n + 1):
            if all(not (_lcm(a,b)==_lcm(a,x)==_lcm(b,x)) for a,b in combinations(selected,2)):
                selected.append(x)
        return selected
    
    # Phase 2: Simulated Annealing to polish
    if time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 2
        # Run SA for a few iterations with different seeds
        num_sa = 3
        sa_time_each = remaining_time / num_sa
        
        for i in range(num_sa):
            if time.time() > deadline - 2:
                break
            rng = random.Random(1000 + i)
            improved_removed = _simulated_annealing(best_removed, triples, elem_to_triples, max_n, rng, sa_time_each)
            size = max_n - len(improved_removed)
            if size > best_size:
                best_size = size
                best_removed = improved_removed
    
    return [x for x in range(1, max_n + 1) if x not in best_removed]

# EVOLVE-BLOCK-END
