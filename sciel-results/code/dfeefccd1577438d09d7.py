"""Optimized solver for large sets avoiding reciprocal Schur triples."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all valid triples (a, b, c) of DISTINCT elements in [1, N]
    such that 1/a = 1/b + 1/c.
    """
    triples = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            if (a * a) % d == 0:
                b = a + d
                c = (a * b) // d
                if 1 <= c <= N:
                    if c != a and c != b:
                        if b != c:
                            triples.append((a, b, c))
    return triples

def _build_index(N=180, triples=None):
    if triples is None:
        triples = _precompute_triples(N)
    
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
    
    return triples, elem_to_triples

def _remove_greedy_degree(N=180, triples=None, elem_to_triples=None, seed=None, deadline=None):
    """
    Greedy removal based on element degree in the active hypergraph.
    """
    if seed is not None:
        random.seed(seed)
        
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    present = [True] * (N + 1)
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = len(elem_to_triples[i])
        
    triple_counts = [3] * len(triples)
    active_list = list(range(len(triples)))
    
    while active_list:
        if deadline and time.time() > deadline:
            break
            
        pos = random.randint(0, len(active_list) - 1)
        t_idx = active_list[pos]
        a, b, c = triples[t_idx]
        
        max_deg = -1
        best_elem = -1
        for x in (a, b, c):
            if degrees[x] > max_deg:
                max_deg = degrees[x]
                best_elem = x
            elif degrees[x] == max_deg:
                if random.random() < 0.5:
                    best_elem = x
                    max_deg = degrees[x]
                    
        present[best_elem] = False
        
        for t_idx2 in elem_to_triples[best_elem]:
            if triple_counts[t_idx2] > 0:
                triple_counts[t_idx2] -= 1
                if triple_counts[t_idx2] == 2:
                    try:
                        active_list.remove(t_idx2)
                    except ValueError:
                        pass
                    a2, b2, c2 = triples[t_idx2]
                    for y in (a2, b2, c2):
                        if y != best_elem:
                            degrees[y] -= 1
                            
    result = [i for i in range(1, N + 1) if present[i]]
    return result

def _remove_random_greedy(N=180, triples=None, elem_to_triples=None, seed=None, deadline=None):
    """
    Randomized greedy hitting set removal.
    """
    if seed is not None:
        random.seed(seed)
        
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    present = [True] * (N + 1)
    triple_counts = [3] * len(triples)
    active_list = list(range(len(triples)))
    
    while active_list:
        if deadline and time.time() > deadline:
            break
        pos = random.randint(0, len(active_list) - 1)
        t_idx = active_list[pos]
        a, b, c = triples[t_idx]
        x_to_remove = random.choice([a, b, c])
        present[x_to_remove] = False
        
        for t_idx2 in elem_to_triples[x_to_remove]:
            if triple_counts[t_idx2] > 0:
                triple_counts[t_idx2] -= 1
                if triple_counts[t_idx2] == 2:
                    try:
                        active_list.remove(t_idx2)
                    except ValueError:
                        pass
                        
    result = [i for i in range(1, N + 1) if present[i]]
    return result

def _local_search(initial_set, N=180, triples=None, elem_to_triples=None, max_iters=1000, deadline=None):
    """
    Try to add elements back to the set if they don't create a violation.
    """
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    present = set(initial_set)
    candidates = set(range(1, N + 1)) - present
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for x in cand_list:
            if deadline and time.time() > deadline:
                break
            if x in present:
                continue
                
            valid = True
            for t_idx in elem_to_triples[x]:
                a, b, c = triples[t_idx]
                others = [y for y in (a, b, c) if y != x]
                if others[0] in present and others[1] in present:
                    valid = False
                    break
                    
            if valid:
                present.add(x)
                candidates.remove(x)
                improved = True
                
    return list(present)

def _greedy_add(N=180, triples=None, elem_to_triples=None, seed=None, deadline=None):
    """
    Greedy addition: start with empty set, add elements one by one if valid.
    """
    if seed is not None:
        random.seed(seed)
        
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    present = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        valid = True
        for t_idx in elem_to_triples[x]:
            a, b, c = triples[t_idx]
            others = [y for y in (a, b, c) if y != x]
            if others[0] in present and others[1] in present:
                valid = False
                break
        if valid:
            present.add(x)
            
    return list(present)

def construct():
    N = 180
    start_time = time.time()
    deadline = start_time + 85.0
    
    triples, elem_to_triples = _build_index(N)
    
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Strategy 1: Degree-based Greedy Removal
    num_degree_runs = 30
    for i in range(num_degree_runs):
        if time.time() > deadline:
            break
        seed = i
        s = _remove_greedy_degree(N, triples, elem_to_triples, seed=seed, deadline=deadline)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
        if time.time() < deadline:
            s2 = _local_search(s, N, triples, elem_to_triples, max_iters=200, deadline=deadline)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)

    # Strategy 2: Random Greedy Removal
    num_removal_runs = 30
    for i in range(num_removal_runs):
        if time.time() > deadline:
            break
        seed = 1000 + i
        s = _remove_random_greedy(N, triples, elem_to_triples, seed=seed, deadline=deadline)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
        if time.time() < deadline:
            s2 = _local_search(s, N, triples, elem_to_triples, max_iters=200, deadline=deadline)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)

    # Strategy 3: Greedy Add
    num_add_runs = 20
    for i in range(num_add_runs):
        if time.time() > deadline:
            break
        seed = 2000 + i
        s = _greedy_add(N, triples, elem_to_triples, seed=seed, deadline=deadline)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
        if time.time() < deadline:
            s2 = _local_search(s, N, triples, elem_to_triples, max_iters=200, deadline=deadline)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)

    # Final Local Search
    if time.time() < deadline:
        s_final = _local_search(list(best_set), N, triples, elem_to_triples, max_iters=1000, deadline=deadline)
        if len(s_final) > best_size:
            best_set = set(s_final)
            
    return list(best_set)

# EVOLVE-BLOCK-END
