"""Strategy: Greedy removal + multi-restart + local search for max set avoiding reciprocal Schur triples."""
import time
import random

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if (a * b) % diff != 0:
                continue
            c = (a * b) // diff
            if c > n_max or c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _greedy_removal(triples, elem_to_triples, n_max, deadline, tie_break='max'):
    """Greedy: start with all, remove highest-conflict element iteratively."""
    S = set(range(1, n_max + 1))
    triple_incount = [3] * len(triples)
    elem_conflict = {i: 0 for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_conflict[a] += 1
        elem_conflict[b] += 1
        elem_conflict[c] += 1
    total_conflicts = len(triples)
    
    while total_conflicts > 0:
        if time.time() > deadline:
            break
        if tie_break == 'max':
            best_elem = max(S, key=lambda x: elem_conflict[x])
        elif tie_break == 'min':
            max_conf = max(elem_conflict[x] for x in S)
            best_elem = min(x for x in S if elem_conflict[x] == max_conf)
        elif tie_break == 'random':
            max_conf = max(elem_conflict[x] for x in S)
            candidates = [x for x in S if elem_conflict[x] == max_conf]
            best_elem = random.choice(candidates)
        else:
            best_elem = max(S, key=lambda x: elem_conflict[x])
        
        if elem_conflict[best_elem] == 0:
            break
        S.remove(best_elem)
        for tidx in elem_to_triples[best_elem]:
            if triple_incount[tidx] == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        elem_conflict[x] -= 1
            triple_incount[tidx] -= 1
    return S

def _local_search(triples, elem_to_triples, n_max, start_set, deadline, max_iter=10000):
    """Local search: try adding removed elements, and 2-opt swaps."""
    S = set(start_set)
    removed = set(range(1, n_max + 1)) - S
    
    def is_valid_add(elem, current_S):
        """Check if adding elem to current_S creates any forbidden triple."""
        for tidx in elem_to_triples[elem]:
            a, b, c = triples[tidx]
            others = [x for x in (a, b, c) if x != elem]
            if all(x in current_S for x in others):
                return False
        return True
    
    best_size = len(S)
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try adding a removed element
        found_add = False
        candidates = list(removed)
        random.shuffle(candidates)
        for elem in candidates[:min(100, len(candidates))]:
            if is_valid_add(elem, S):
                S.add(elem)
                removed.discard(elem)
                best_size = max(best_size, len(S))
                found_add = True
                no_improve_count = 0
                break
        
        if not found_add:
            no_improve_count += 1
            if no_improve_count > 200:
                break
            
            # Try 2-opt: remove one element from S, then try to add two from removed
            s_elements = list(S)
            random.shuffle(s_elements)
            found_swap = False
            for rem_elem in s_elements[:min(30, len(s_elements))]:
                S_test = S - {rem_elem}
                removed_test = removed | {rem_elem}
                
                # Try adding one element
                candidates2 = list(removed_test)
                random.shuffle(candidates2)
                for add_elem in candidates2[:min(50, len(candidates2))]:
                    if is_valid_add(add_elem, S_test):
                        S_new = S_test | {add_elem}
                        # Try adding one more
                        candidates3 = list(removed_test - {add_elem})
                        random.shuffle(candidates3)
                        for add_elem2 in candidates3[:min(50, len(candidates3))]:
                            if is_valid_add(add_elem2, S_new):
                                S_final = S_new | {add_elem2}
                                if len(S_final) > best_size:
                                    S = S_final
                                    removed = set(range(1, n_max + 1)) - S
                                    best_size = len(S)
                                    found_swap = True
                                    no_improve_count = 0
                                    break
                        if found_swap:
                            break
                if found_swap:
                    break
            if found_swap:
                continue
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n_max = 180
    random.seed(42)
    
    triples = _build_triples(n_max)
    
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy removal (max conflict first)
    S1 = _greedy_removal(triples, elem_to_triples, n_max, deadline, tie_break='max')
    best_set = S1
    best_size = len(S1)
    
    # Strategy 2: Greedy removal (min value tie-break)
    if time.time() < deadline - 10:
        S2 = _greedy_removal(triples, elem_to_triples, n_max, deadline, tie_break='min')
        if len(S2) > best_size:
            best_set = S2
            best_size = len(S2)
    
    # Strategy 3: Greedy removal (random tie-break)
    if time.time() < deadline - 10:
        S3 = _greedy_removal(triples, elem_to_triples, n_max, deadline, tie_break='random')
        if len(S3) > best_size:
            best_set = S3
            best_size = len(S3)
    
    # Strategy 4: Another random tie-break
    if time.time() < deadline - 10:
        S4 = _greedy_removal(triples, elem_to_triples, n_max, deadline, tie_break='random')
        if len(S4) > best_size:
            best_set = S4
            best_size = len(S4)
    
    # Local search on the best solution
    if time.time() < deadline - 3:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline, max_iter=5000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    # One more local search pass if time allows
    if time.time() < deadline - 2:
        ls_set2 = _local_search(triples, elem_to_triples, n_max, best_set, deadline, max_iter=3000)
        if len(ls_set2) > best_size:
            best_set = ls_set2
            best_size = len(ls_set2)
    
    if not best_set:
        best_set = set(range(91, 181))
    
    return list(best_set)
