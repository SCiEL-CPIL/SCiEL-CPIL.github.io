import time
import random

N = 256
DEADLINE = 58.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    
    # Precompute 3-APs: (a, a+d, a+2d)
    three_aps = []
    three_aps_by_elem = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            three_aps_by_elem[a].append(idx)
            three_aps_by_elem[b].append(idx)
            three_aps_by_elem[c].append(idx)
            
    # Precompute 4-APs: (a, a+d, a+2d, a+3d)
    four_aps_by_v = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            for v in [a, b, c, e]:
                others = [x for x in [a, b, c, e] if x != v]
                four_aps_by_v[v].append(others)
                
    # Helper to check if adding v creates a 4-AP given current set
    def has_4ap_with_v(current_set, v):
        for others in four_aps_by_v[v]:
            if all(o in current_set for o in others):
                return True
        return False
    
    # Helper to check if swap u->v is valid
    def is_valid_swap(current_set, u, v):
        # Check all 4-APs containing v
        for others in four_aps_by_v[v]:
            # Condition: all elements in 'others' (except possibly u) are in current_set \ {u}
            # i.e., all(o in current_set for o in others if o != u)
            valid = True
            for o in others:
                if o == u:
                    continue
                if o not in current_set:
                    valid = False
                    break
            if valid:
                return False
        return True
    
    # Count 3-APs in a set
    def count_3aps(current_set):
        count = 0
        for a, b, c in three_aps:
            if a in current_set and b in current_set and c in current_set:
                count += 1
        return count
    
    # Initial greedy set
    current_set = set()
    for x in range(1, N+1):
        if not has_4ap_with_v(current_set, x):
            current_set.add(x)
            
    current_score = count_3aps(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            break
            
        # Pick random u from current_set, v from complement
        elements = list(current_set)
        non_elements = [i for i in range(1, N+1) if i not in current_set]
        
        if not elements or not non_elements:
            break
            
        u = random.choice(elements)
        v = random.choice(non_elements)
        
        if not is_valid_swap(current_set, u, v):
            continue
            
        # Compute score change: remove 3-APs containing u, add 3-APs containing v
        # A 3-AP is counted if all three elements are in the set
        # When we remove u and add v:
        # - 3-APs that contain u and at least one of the other two was in set: lose them
        #   Actually: a 3-AP with u is counted iff the other two are in set.
        #   After swap: u is gone, so any 3-AP containing u is no longer counted.
        #   But 3-APs not containing u but containing v might now be counted.
        #   Also, 3-APs containing both u and v: were they counted before? Only if the third was in set.
        #   After swap, they're not counted (u gone).
        
        # Delta = (new count) - (old count)
        # New count = old count - (3-APs containing u that were fully in old set)
        #              + (3-APs containing v that are fully in new set)
        # A 3-AP containing u was fully in old set iff the other two are in old set.
        # A 3-AP containing v is fully in new set iff the other two are in new_set = old_set - {u} + {v}.
        
        delta = 0
        
        # Subtract 3-APs containing u that were fully present
        for ap_idx in three_aps_by_elem[u]:
            a, b, c = three_aps[ap_idx]
            others = [x for x in (a, b, c) if x != u]
            if all(o in current_set for o in others):
                delta -= 1
        
        # Add 3-APs containing v that will be fully present after swap
        new_set_preview = (current_set - {u}) | {v}
        for ap_idx in three_aps_by_elem[v]:
            a, b, c = three_aps[ap_idx]
            others = [x for x in (a, b, c) if x != v]
            if all(o in new_set_preview for o in others):
                delta += 1
        
        new_score = current_score + delta
        
        # Acceptance criterion
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_set.discard(u)
            current_set.add(v)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        
        T *= cooling_rate
    
    return sorted(best_set)
