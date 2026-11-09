import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 180
    elements = list(range(1, N+1))
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    # c = ab/(a+b), need a+b | ab
    forbidden = []  # list of (a, b, c) triples
    triple_set = set()
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            s = a + b
            p = a * b
            if p % s == 0:
                c = p // s
                if c > b and c <= N:
                    triple_set.add((a, b, c))
                    forbidden.append((a, b, c))
                elif c > a and c < b and c <= N:
                    # c is between a and b
                    triple_set.add((a, c, b))
                    forbidden.append((a, c, b))
                elif c < a and c <= N:
                    triple_set.add((c, a, b))
                    forbidden.append((c, a, b))
    
    # Build conflict graph: for each element, which other elements it conflicts with
    # Actually, build for each element, the set of triples it's in
    from collections import defaultdict
    element_triples = defaultdict(list)
    for (a, b, c) in triple_set:
        element_triples[a].append((b, c))
        element_triples[b].append((a, c))
        element_triples[c].append((a, b))
    
    # Start with all elements
    in_set = [True] * (N+1)  # 1-indexed
    
    # Greedy removal: while conflicts exist, remove the element in most conflicts
    def count_conflicts(in_set):
        conflicts = 0
        for (a, b, c) in triple_set:
            if in_set[a] and in_set[b] and in_set[c]:
                conflicts += 1
        return conflicts
    
    # Iterative removal
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        # Find all active conflicts
        active_conflicts = []
        for (a, b, c) in triple_set:
            if in_set[a] and in_set[b] and in_set[c]:
                active_conflicts.append((a, b, c))
        
        if not active_conflicts:
            break
        
        # Count how many conflicts each element is in
        conflict_count = defaultdict(int)
        for (a, b, c) in active_conflicts:
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
        
        # Remove the element with most conflicts
        max_conf = -1
        max_elem = -1
        for elem in range(1, N+1):
            if in_set(elem) if callable(in_set) else in_set[elem]:
                pass
        for elem in range(1, N+1):
            if in_set[elem]:
                if conflict_count[elem] > max_conf:
                    max_conf = conflict_count[elem]
                    max_elem = elem
        
        if max_elem == -1:
            break
        in_set[max_elem] = False
    
    # Now try to add back removed elements
    removed = [i for i in range(1, N+1) if not in_set[i]]
    
    def is_valid_addition(elem, current_set):
        """Check if adding elem to current_set creates any forbidden triple"""
        for (other1, other2) in element_triples[elem]:
            if current_set[other1] and current_set[other2]:
                return False
        return True
    
    # Try adding back in random order, multiple passes
    best_size = sum(in_set)
    best_set = in_set.copy()
    
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Shuffle removed elements
        shuffled = removed[:]
        random.shuffle(shuffled)
        
        current_set = best_set.copy()
        current_size = best_size
        
        for elem in shuffled:
            if not current_set[elem]:
                if is_valid_addition(elem, current_set):
                    current_set[elem] = True
                    current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set.copy()
            # Update removed list
            removed = [i for i in range(1, N+1) if not best_set[i]]
    
    # Local search: try swapping
    for trial in range(200):
        if time.time() > deadline - 3:
            break
        
        improved = False
        # Try removing one and adding another
        in_set_elems = [i for i in range(1, N+1) if best_set[i]]
        out_set_elems = [i for i in range(1, N+1) if not best_set[i]]
        
        if not out_set_elems:
            break
        
        random.shuffle(in_set_elems)
        random.shuffle(out_set_elems)
        
        for rem in in_set_elems[:20]:
            for add in out_set_elems[:20]:
                if time.time() > deadline - 3:
                    break
                # Try the swap
                if not best_set[rem] or best_set[add]:
                    continue
                # Check if removing rem and adding add is valid
                best_set[rem] = False
                best_set[add] = True
                # Check validity
                valid = True
                for (a, b, c) in triple_set:
                    if best_set[a] and best_set[b] and best_set[c]:
                        valid = False
                        break
                if valid:
                    improved = True
                    break
                else:
                    best_set[rem] = True
                    best_set[add] = False
            if improved:
                break
        if not improved:
            break
    
    result = [i for i in range(1, N+1) if best_set[i]]
    return result
