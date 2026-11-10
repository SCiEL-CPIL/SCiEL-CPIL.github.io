# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    
    Uses:
    1. Deterministic greedy initialization
    2. Multiple random greedy initializations
    3. Exhaustive local search (try all single additions)
    4. Simulated annealing with random restarts
    5. Pair-swap local search
    6. Final exhaustive polish
    
    The forbidden differences are the nonzero quadratic residues modulo 205.
    """
    m = 205
    deadline = time.time() + 590  # Safety margin before 600s kill
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # Precompute conflict matrix: conflict[i][j] = True if i and j cannot both be in the set
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def is_valid(selected_set):
        """Check if a set is valid (no forbidden differences)."""
        arr = sorted(selected_set)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if conflict[arr[i], arr[j]]:
                    return False
        return True
    
    def greedy_init_random(seed):
        """Greedy initialization with random ordering."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected = set()
        for candidate in candidates:
            ok = True
            for s in selected:
                if conflict[candidate, s]:
                    ok = False
                    break
            if ok:
                selected.add(candidate)
        return selected
    
    def can_add(selected_set, candidate):
        """Check if candidate can be added to selected_set."""
        for s in selected_set:
            if conflict[candidate, s]:
                return False
        return True
    
    def exhaustive_local_search(initial_set, time_limit=10.0):
        """Exhaustive local search: repeatedly try all possible single additions until no more can be added."""
        current = set(initial_set)
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for candidate in range(m):
                if candidate in current:
                    continue
                if can_add(current, candidate):
                    current.add(candidate)
                    improved = True
                    break
        
        return current
    
    def pair_swap_search(initial_set, time_limit=30.0):
        """Try all pair swaps (remove two, add two different elements)."""
        current = set(initial_set)
        best = set(initial_set)
        start_time = time.time()
        
        while (time.time() - start_time) < time_limit:
            improved = False
            cur_list = list(current)
            n = len(cur_list)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if (time.time() - start_time) >= time_limit:
                        break
                    # Try removing cur_list[i] and cur_list[j]
                    temp = set(current)
                    temp.discard(cur_list[i])
                    temp.discard(cur_list[j])
                    
                    # Try adding two new elements
                    added_any = False
                    for a in range(m):
                        if a in temp:
                            continue
                        if can_add(temp, a):
                            temp.add(a)
                            added_any = True
                            break
                        # Also try without a, see if we can add two
                        for b in range(a + 1, m):
                            if b in temp:
                                continue
                            if can_add(temp, b):
                                temp.add(b)
                                added_any = True
                                break
                            # Try adding just one more
                            for c in range(b + 1, m):
                                if c in temp:
                                    continue
                                if can_add(temp, c):
                                    temp.add(c)
                                    added_any = True
                                    break
                            if added_any:
                                break
                        if added_any:
                            break
                    
                    if added_any and len(temp) > len(best):
                        best = set(temp)
                        current = set(temp)
                        improved = True
            
            if not improved:
                break
        
        return best
    
    def simulated_annealing(initial_set, seed, time_limit=60.0):
        """Simulated annealing to improve the set."""
        rng = random.Random(seed)
        current = set(initial_set)
        best = set(initial_set)
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.001
        cooling = 0.997
        T = T_start
        
        start_time = time.time()
        
        while (time.time() - start_time) < time_limit:
            if T < T_end:
                break
            
            # Generate a random move
            move_type = rng.choice(['add', 'remove', 'swap', 'swap2'])
            
            if move_type == 'add':
                # Try to add a random element
                candidates = [i for i in range(m) if i not in current]
                if not candidates:
                    T *= cooling
                    continue
                candidate = rng.choice(candidates)
                if can_add(current, candidate):
                    new_set = set(current)
                    new_set.add(candidate)
                    current = new_set
                    if len(current) > len(best):
                        best = set(current)
                else:
                    # Metropolis acceptance for invalid moves
                    if rng.random() < T * 0.1:
                        current.add(candidate)  # Temporarily accept invalid state
            
            elif move_type == 'remove':
                # Try to remove a random element
                if not current:
                    T *= cooling
                    continue
                candidate = rng.choice(list(current))
                new_set = set(current)
                new_set.discard(candidate)
                # Accept with some probability to escape local optima
                if rng.random() < 0.15:
                    current = new_set
            
            elif move_type == 'swap':
                # Try to swap: remove one, add another
                if not current:
                    T *= cooling
                    continue
                remove_candidate = rng.choice(list(current))
                add_candidates = [i for i in range(m) if i not in current]
                if not add_candidates:
                    T *= cooling
                    continue
                add_candidate = rng.choice(add_candidates)
                
                new_set = set(current)
                new_set.discard(remove_candidate)
                new_set.add(add_candidate)
                
                # Check if new set is valid
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a != b and conflict[a, b]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current = new_set
                    if len(current) > len(best):
                        best = set(current)
            
            elif move_type == 'swap2':
                # Remove two, add two
                if len(current) < 2:
                    T *= cooling
                    continue
                rem1, rem2 = rng.sample(list(current), 2)
                add_candidates = [i for i in range(m) if i not in current]
                if len(add_candidates) < 2:
                    T *= cooling
                    continue
                add1, add2 = rng.sample(add_candidates, 2)
                
                new_set = set(current)
                new_set.discard(rem1)
                new_set.discard(rem2)
                new_set.add(add1)
                new_set.add(add2)
                
                # Check if new set is valid
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a != b and conflict[a, b]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current = new_set
                    if len(current) > len(best):
                        best = set(current)
            
            T *= cooling
        
        return best
    
    # Start with deterministic greedy
    best_set = greedy_init_random(0)
    best_set = exhaustive_local_search(best_set, time_limit=5.0)
    
    # Try multiple random initializations
    for seed in range(100):
        if time.time() > deadline - 60:
            break
        init_set = greedy_init_random(seed)
        init_set = exhaustive_local_search(init_set, time_limit=2.0)
        if len(init_set) > len(best_set):
            best_set = init_set
    
    # Apply simulated annealing with multiple restarts
    for seed in range(30):
        if time.time() > deadline - 30:
            break
        # Start from best_set or a random perturbation
        if seed % 3 == 0:
            start_set = set(best_set)
        elif seed % 3 == 1:
            start_set = greedy_init_random(seed + 1000)
            start_set = exhaustive_local_search(start_set, time_limit=1.0)
        else:
            # Perturb best_set
            start_set = set(best_set)
            rng = random.Random(seed + 2000)
            for _ in range(rng.randint(1, 3)):
                if start_set:
                    to_remove = rng.choice(list(start_set))
                    start_set.discard(to_remove)
                    # Try to add something
                    for cand in range(m):
                        if cand not in start_set and can_add(start_set, cand):
                            start_set.add(cand)
                            break
        
        remaining_time = max(10, deadline - 30 - (time.time() - start_time if 'start_time' in dir() else 0))
        sa_result = simulated_annealing(start_set, seed, time_limit=min(40, remaining_time))
        sa_result = exhaustive_local_search(sa_result, time_limit=2.0)
        if len(sa_result) > len(best_set):
            best_set = sa_result
    
    # Final pair-swap search
    if time.time() < deadline - 15:
        swap_result = pair_swap_search(best_set, time_limit=20.0)
        swap_result = exhaustive_local_search(swap_result, time_limit=3.0)
        if len(swap_result) > len(best_set):
            best_set = swap_result
    
    # Final exhaustive polish
    final_set = exhaustive_local_search(best_set, time_limit=5.0)
    if len(final_set) > len(best_set):
        best_set = final_set
    
    # Convert to sorted array
    result = np.array(sorted(best_set), dtype=np.int64)
    
    # Validate the result
    if not is_valid(set(result.tolist())):
        # Fallback to a known valid set
        result = np.array(sorted(greedy_init_random(0)), dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
