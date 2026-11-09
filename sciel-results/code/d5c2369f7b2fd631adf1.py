import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 180
    
    # Generate all forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c => c = a*b/(b-a)
    # Or equivalently: a = k*u*v, b = k*u*(u+v), c = k*v*(u+v) for coprime u,v
    
    triples = set()
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if math.gcd(u, v) != 1:
                continue
            k_max = min(N // (u * v), N // (u * (u + v)), N // (v * (u + v)))
            for k in range(1, k_max + 1):
                a = k * u * v
                b = k * u * (u + v)
                c = k * v * (u + v)
                if a <= N and b <= N and c <= N and a != b and a != c and b != c:
                    triples.add((a, b, c))
    
    triples = list(triples)
    
    # Build adjacency: for each vertex, which triples contain it
    vertex_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for v in t:
            vertex_triples[v].append(idx)
    
    # Greedy construction: add vertices in order of increasing constraint
    # A vertex is "safe" to add if it doesn't complete any triple with already-selected vertices
    selected = set()
    
    # Sort vertices by number of triples they participate in (add less-constrained first)
    vertex_order = sorted(range(1, N + 1), key=lambda v: len(vertex_triples[v]))
    
    for v in vertex_order:
        # Check if adding v creates any forbidden triple
        can_add = True
        for t_idx in vertex_triples[v]:
            t = triples[t_idx]
            # Check if the other two elements are already in selected
            others = [x for x in t if x != v]
            if all(o in selected for o in others):
                can_add = False
                break
        if can_add:
            selected.add(v)
    
    # Local search: try to improve by swapping
    def is_valid_addition(selected_set, v):
        """Check if adding v to selected_set creates no forbidden triple."""
        for t_idx in vertex_triples[v]:
            t = triples[t_idx]
            others = [x for x in t if x != v]
            if all(o in selected_set for o in others):
                return False
        return True
    
    def try_improve(selected_set, time_limit):
        """Try local search improvements."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            selected_list = list(selected_set)
            random.shuffle(selected_list)
            
            for v_remove in selected_list:
                if time.time() >= time_limit:
                    break
                
                # Try removing v_remove and adding some other vertex
                new_selected = selected_set - {v_remove}
                
                # Find candidates to add
                candidates = [v for v in range(1, N + 1) if v not in new_selected]
                random.shuffle(candidates)
                
                for v_add in candidates:
                    if v_add == v_remove:
                        continue
                    if is_valid_addition(new_selected, v_add):
                        # This swap is valid; accept it (since we're removing 1 and adding 1, size stays same)
                        # But we want to try to INCREASE size, so this is neutral
                        # Actually, let's try 1-remove + 2-add or just track best
                        pass
            
            # Try 1-remove + 2-add moves
            for v_remove in selected_list:
                if time.time() >= time_limit:
                    break
                
                new_selected = selected_set - {v_remove}
                
                # Find pairs of candidates that can both be added
                candidates = [v for v in range(1, N + 1) if v not in new_selected]
                random.shuffle(candidates)
                
                found_pair = False
                for i in range(len(candidates)):
                    if time.time() >= time_limit:
                        break
                    v1 = candidates[i]
                    if not is_valid_addition(new_selected, v1):
                        continue
                    
                    temp_selected = new_selected | {v1}
                    for j in range(i + 1, len(candidates)):
                        v2 = candidates[j]
                        if is_valid_addition(temp_selected, v2):
                            # Great! We can add both
                            selected_set = temp_selected | {v2}
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    break
        
        return selected_set
    
    # Run local search for a while
    local_search_deadline = deadline - 5  # Leave 5s for final processing
    selected = try_improve(selected, local_search_deadline)
    
    # Final validation
    final_set = set(selected)
    for t in triples:
        if all(v in final_set for v in t):
            # This shouldn't happen, but remove one element to fix
            final_set.remove(t[0])
            break
    
    return sorted(list(final_set))
