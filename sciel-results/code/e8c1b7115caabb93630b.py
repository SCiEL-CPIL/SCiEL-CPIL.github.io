"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 512
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d)
    ap_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Build index: for each element, which APs contain it
    elem_to_aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(ap_list):
        for x in ap:
            elem_to_aps[x].append(idx)
    
    # Greedy construction
    def greedy():
        selected = set()
        for x in range(1, N + 1):
            if all(not all(x - j * d in selected for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
                selected.add(x)
        return selected
    
    # Check if adding element x to set S creates a 4-AP
    def is_safe(x, S):
        # Check all APs containing x
        for ap in elem_to_aps[x]:
            a, b, c, d = ap
            others = [y for y in ap if y != x]
            if all(y in S for y in others):
                return False
        return True
    
    # Check if set S has any 4-AP
    def has_ap(S):
        for ap in ap_list:
            if all(x in S for x in ap):
                return True
        return False
    
    # Local search with 2-exchanges
    def local_search(S, time_limit):
        S = set(S)
        best = len(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try 1-in, 1-out exchanges
            for out_elem in list(S):
                if time.time() > time_limit:
                    break
                S.remove(out_elem)
                # Try adding each non-selected element
                for in_elem in range(1, N + 1):
                    if in_elem in S:
                        continue
                    if is_safe(in_elem, S):
                        S.add(in_elem)
                        if len(S) > best:
                            best = len(S)
                            improved = True
                        S.remove(in_elem)
                S.add(out_elem)
            
            # Try 2-in, 2-out exchanges (sampled for speed)
            if not improved and time.time() < time_limit:
                in_candidates = [x for x in range(1, N + 1) if x not in S]
                out_candidates = list(S)
                random.shuffle(in_candidates)
                random.shuffle(out_candidates)
                
                max_pairs = 2000
                for i in range(min(len(out_candidates), 50)):
                    if time.time() > time_limit:
                        break
                    out1 = out_candidates[i]
                    for j in range(i + 1, min(len(out_candidates), 50)):
                        out2 = out_candidates[j]
                        S.remove(out1)
                        S.remove(out2)
                        
                        for in1 in in_candidates[:100]:
                            if time.time() > time_limit:
                                break
                            if not is_safe(in1, S):
                                continue
                            S.add(in1)
                            for in2 in in_candidates[:100]:
                                if in2 == in1:
                                    continue
                                if not is_safe(in2, S):
                                    continue
                                S.add(in2)
                                if len(S) > best:
                                    best = len(S)
                                    improved = True
                                S.remove(in2)
                            S.remove(in1)
                        
                        S.add(out1)
                        S.add(out2)
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    # Initial greedy
    S = greedy()
    best_set = set(S)
    best_size = len(S)
    
    # Local search
    time_limit = deadline - 5
    S = local_search(S, time_limit)
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Try multiple random restarts if time permits
    if time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            # Random perturbation
            S2 = set(best_set)
            # Remove some random elements
            elems = list(S2)
            random.shuffle(elems)
            for e in elems[:len(elems) // 4]:
                S2.remove(e)
            # Greedy fill
            for x in range(1, N + 1):
                if x not in S2 and is_safe(x, S2):
                    S2.add(x)
            # Local search
            time_limit2 = min(deadline - 5, time.time() + 3)
            S2 = local_search(S2, time_limit2)
            if len(S2) > best_size:
                best_set = set(S2)
                best_size = len(S2)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
