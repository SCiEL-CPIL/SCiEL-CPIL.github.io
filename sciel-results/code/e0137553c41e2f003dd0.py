import time
import math
import random

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    return triples

def _check_valid(S, triples):
    """Check if set S has no valid triple from the precomputed list."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def _check_valid_fast(S_set, conflict_map):
    """Check validity using conflict map for speed."""
    for x in S_set:
        if x in conflict_map:
            for (y, z) in conflict_map[x]:
                if y in S_set and z in S_set:
                    return False
    return True

def _build_conflict_map(triples):
    """Build a map: element -> list of (other1, other2) where {element, other1, other2} forms a triple."""
    conflict_map = {}
    for (a, b, c) in triples:
        for x in (a, b, c):
            others = [y for y in (a, b, c) if y != x]
            if x not in conflict_map:
                conflict_map[x] = []
            conflict_map[x].append(tuple(sorted(others)))
    # Deduplicate
    for x in conflict_map:
        seen = set()
        unique = []
        for pair in conflict_map[x]:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)
        conflict_map[x] = unique
    return conflict_map

def _greedy_add(S, elements_to_try, triples, conflict_map, deadline):
    """Greedily add elements from elements_to_try to S if valid."""
    S_set = set(S)
    for x in elements_to_try:
        if time.time() > deadline:
            break
        if x not in S_set:
            S_set.add(x)
            # Check if adding x creates a conflict
            valid = True
            if x in conflict_map:
                for (y, z) in conflict_map[x]:
                    if y in S_set and z in S_set:
                        valid = False
                        break
            if not valid:
                S_set.remove(x)
    return S_set

def _local_search(S, triples, conflict_map, deadline, max_iters=200):
    """Local search: try removing one and adding another."""
    S_set = set(S)
    current_size = len(S_set)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        best_swap = None
        best_new_set = None
        best_size = current_size
        
        elements = list(S_set)
        random.shuffle(elements)
        
        for x_remove in elements[:20]:  # Limit to first 20 for speed
            if time.time() > deadline:
                break
            
            reduced = S_set - {x_remove}
            
            # Try to add elements not in reduced
            available = [y for y in range(1, 181) if y not in reduced and y != x_remove]
            random.shuffle(available)
            
            for x_add in available[:30]:  # Limit to first 30
                if time.time() > deadline:
                    break
                
                new_set = reduced | {x_add}
                # Quick check
                valid = True
                if x_add in conflict_map:
                    for (y, z) in conflict_map[x_add]:
                        if y in new_set and z in new_set:
                            valid = False
                            break
                if not valid:
                    continue
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_new_set = new_set
                    best_swap = (x_remove, x_add)
        
        if best_new_set is not None:
            S_set = best_new_set
            current_size = best_size
        else:
            break
    
    return S_set

def _simulated_annealing(S, triples, conflict_map, deadline, initial_temp=10.0, cooling=0.95):
    """Simulated annealing to improve the set."""
    S_set = set(S)
    current_size = len(S_set)
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 0.1:
            break
        
        # Generate a random move
        move_type = random.random()
        S_copy = set(S_set)
        
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(1, 181) if x not in S_copy]
            if not candidates:
                continue
            x_add = random.choice(candidates)
            S_copy.add(x_add)
            delta = 1
        elif move_type < 0.8:
            # Remove a random element
            if not S_copy:
                continue
            x_remove = random.choice(list(S_copy))
            S_copy.remove(x_remove)
            delta = -1
        else:
            # Swap: remove one, add one
            if not S_copy:
                continue
            candidates_add = [x for x in range(1, 181) if x not in S_copy]
            if not candidates_add:
                continue
            x_remove = random.choice(list(S_copy))
            x_add = random.choice(candidates_add)
            S_copy.remove(x_remove)
            S_copy.add(x_add)
            delta = 0
        
        # Check validity of new set
        valid = _check_valid_fast(S_copy, conflict_map)
        
        if valid:
            S_set = S_copy
            current_size = len(S_set)
        else:
            # Accept with probability exp(-delta/temp) if delta < 0 (worse)
            if delta > 0:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    S_set = S_copy
                    current_size = len(S_set)
        
        temp *= cooling
    
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    N = 180
    
    triples = _precompute_triples(N)
    conflict_map = _build_conflict_map(triples)
    
    # Strategy 1: Find the largest valid contiguous suffix {K, K+1, ..., 180}
    best_suffix = None
    for K in range(1, N + 1):
        S = set(range(K, N + 1))
        if _check_valid_fast(S, conflict_map):
            best_suffix = S
            break
    
    if best_suffix is None:
        best_suffix = set(range(1, N + 1))
    
    best_set = set(best_suffix)
    best_size = len(best_set)
    K = min(best_suffix) if best_suffix else 1
    
    # Strategy 2: Greedy add smaller elements
    if time.time() < deadline - 10.0:
        candidates = list(range(1, K))
        random.shuffle(candidates)
        best_set = _greedy_add(best_set, candidates, triples, conflict_map, deadline - 10.0)
        best_size = len(best_set)
    
    # Strategy 3: Local search
    if time.time() < deadline - 5.0:
        new_set = _local_search(best_set, triples, conflict_map, deadline - 5.0, max_iters=100)
        if len(new_set) > best_size:
            best_set = new_set
            best_size = len(best_set)
    
    # Strategy 4: Simulated annealing
    if time.time() < deadline - 2.0:
        new_set = _simulated_annealing(best_set, triples, conflict_map, deadline - 2.0, initial_temp=5.0, cooling=0.97)
        if len(new_set) > best_size:
            best_set = new_set
            best_size = len(new_set)
    
    # Strategy 5: Multiple random restarts with greedy
    if time.time() < deadline - 3.0:
        for trial in range(5):
            if time.time() > deadline - 2.0:
                break
            
            S = set()
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            
            for x in elements:
                if time.time() > deadline - 2.0:
                    break
                S.add(x)
                valid = True
                if x in conflict_map:
                    for (y, z) in conflict_map[x]:
                        if y in S and z in S:
                            valid = False
                            break
                if not valid:
                    S.remove(x)
            
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
    
    return list(best_set)

# EVOLVE-BLOCK-END
