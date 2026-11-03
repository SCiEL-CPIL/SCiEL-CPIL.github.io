import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    D = 11
    
    # Generate weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    candidates_list = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            candidates_list.append(v)
            
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5280
    
    # Index vectors by support
    vectors_by_support = {i: [] for i in range(len(supports))}
    for idx, v in enumerate(candidates_list):
        non_zero = tuple(i for i in range(D) if v[i] != 0)
        s_idx = support_to_idx[non_zero]
        vectors_by_support[s_idx].append(idx)
        
    # Build conflict graph
    # Conflict if dot > 2, i.e., if intersection size 3 and signs match on intersection
    # For weight-4 vectors: dot = |intersection| if signs match on intersection
    # dot > 2 means intersection size >= 3 and signs match on intersection
    
    # Precompute support pairs with intersection size 3
    num_supports = len(supports)
    support_pairs = []
    for i in range(num_supports):
        s1 = set(supports[i])
        for j in range(i + 1, num_supports):
            s2 = set(supports[j])
            if len(s1 & s2) == 3:
                support_pairs.append((i, j))
                
    # Build neighbor lists
    neighbors = [set() for _ in range(n)]
    
    for s1_idx, s2_idx in support_pairs:
        s1 = set(supports[s1_idx])
        s2 = set(supports[s2_idx])
        intersection = s1 & s2
        inter_list = list(intersection)
        
        vecs1 = vectors_by_support[s1_idx]
        vecs2 = vectors_by_support[s2_idx]
        
        # Group vecs2 by sign pattern on intersection
        vecs2_by_pattern = {}
        for v2_idx in vecs2:
            v2 = candidates_list[v2_idx]
            pat = tuple(v2[pos] for pos in inter_list)
            if pat not in vecs2_by_pattern:
                vecs2_by_pattern[pat] = []
            vecs2_by_pattern[pat].append(v2_idx)
            
        for v1_idx in vecs1:
            v1 = candidates_list[v1_idx]
            pat1 = tuple(v1[pos] for pos in inter_list)
            if pat1 in vecs2_by_pattern:
                for v2_idx in vecs2_by_pattern[pat1]:
                    neighbors[v1_idx].add(v2_idx)
                    neighbors[v2_idx].add(v1_idx)
    
    # Convert to lists for faster iteration
    neighbors_list = [list(neighbors[i]) for i in range(n)]
    
    # Convert neighbors to sets for O(1) membership testing in local search
    neighbors_set = [set(neighbors_list[i]) for i in range(n)]
    
    def greedy_mis(order, time_limit):
        selected = []
        selected_set = set()
        available = set(range(n))
        
        for idx in order:
            if time.time() > time_limit:
                break
            if idx not in available:
                continue
            selected.append(idx)
            selected_set.add(idx)
            available.discard(idx)
            for nb in neighbors_list[idx]:
                available.discard(nb)
                
        return selected, selected_set, available
    
    def local_search(selected, selected_set, available, time_limit):
        # conflict_count[j] = number of neighbors of j in selected
        conflict_count = np.zeros(n, dtype=np.int32)
        for s in selected:
            for nb in neighbors_list[s]:
                conflict_count[nb] += 1
                
        # Addable nodes: not in selected, conflict_count == 0
        addable = [i for i in available if conflict_count[i] == 0]
        
        # Greedy add phase
        while addable and time.time() < time_limit:
            idx = addable.pop()
            selected.append(idx)
            selected_set.add(idx)
            available.discard(idx)
            for nb in neighbors_list[idx]:
                if nb in available:
                    conflict_count[nb] += 1
                    if conflict_count[nb] == 1:
                        try:
                            addable.remove(nb)
                        except ValueError:
                            pass
        
        # 2-for-1 swap local search
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(selected)
            
            for s in selected:
                if time.time() > time_limit:
                    break
                    
                # Find candidates u that conflict only with s (conflict_count[u] == 1 and s in neighbors[u])
                candidates_u = []
                for u in range(n):
                    if u in selected_set:
                        continue
                    if conflict_count[u] == 1 and s in neighbors_set[u]:
                        candidates_u.append(u)
                        
                if len(candidates_u) < 2:
                    continue
                
                # Try to find a compatible pair (u, v) in candidates_u
                random.shuffle(candidates_u)
                found = False
                max_tries = min(50, len(candidates_u))
                for i in range(max_tries):
                    u = candidates_u[i]
                    for j in range(i + 1, max_tries):
                        v = candidates_u[j]
                        if v not in neighbors_set[u]:
                            # Do the swap: remove s, add u and v
                            selected.remove(s)
                            selected_set.discard(s)
                            
                            selected.append(u)
                            selected_set.add(u)
                            selected.append(v)
                            selected_set.add(v)
                            
                            # Recompute conflict_count
                            conflict_count = np.zeros(n, dtype=np.int32)
                            for sel in selected:
                                for nb in neighbors_list[sel]:
                                    conflict_count[nb] += 1
                                    
                            # Rebuild available and addable
                            available = set(range(n)) - selected_set
                            addable = [i for i in available if conflict_count[i] == 0]
                            
                            # Greedy add again
                            while addable and time.time() < time_limit:
                                idx = addable.pop()
                                selected.append(idx)
                                selected_set.add(idx)
                                available.discard(idx)
                                for nb in neighbors_list[idx]:
                                    if nb in available:
                                        conflict_count[nb] += 1
                                        if conflict_count[nb] == 1:
                                            try:
                                                addable.remove(nb)
                                            except ValueError:
                                                pass
                            
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
                    
        return selected, selected_set, available
    
    # Run multiple restarts
    best_size = 0
    best_result = []
    
    random.seed(42)
    
    num_restarts = 50
    for trial in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Random ordering
        order = list(range(n))
        random.shuffle(order)
        
        # Greedy
        time_limit_inner = deadline - 5.0
        sel, sel_set, avail = greedy_mis(order, time_limit_inner)
        
        # Local search
        sel, sel_set, avail = local_search(sel, sel_set, avail, time_limit_inner)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = list(sel)
            
    if not best_result:
        return candidates[0].reshape(1, -1)
        
    return candidates[np.array(best_result, dtype=np.int64)].astype(np.int64)
