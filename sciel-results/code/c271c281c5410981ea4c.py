import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4. Distance to Type D (norm^2=4) is 8-2*dot.
    # If dot <= 2, dist^2 >= 4.
    # Max dot between A (one 2) and D (four 1s) is 2 (if the 2 is in D's support).
    # So Type A is always compatible with Type D.
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4.
    # Compatibility condition for two Type D vectors u, v:
    # dist^2 = 8 - 2 dot(u,v) >= 4  =>  dot(u,v) <= 2.
    # dot(u,v) = sum(u_i v_i).
    # If supports are disjoint, dot=0.
    # If intersection size k, max dot is k.
    # So we need intersection size <= 2 for safety?
    # If intersection size 3, we can choose signs such that dot <= 2.
    # Example: u=(1,1,1,0...), v=(1,1,-1,0...). dot=1+1-1=1.
    # So supports with intersection 3 are NOT strictly forbidden, but we must choose signs carefully.
    # However, the current program uses a simpler model:
    # It finds an independent set of supports where intersection <= 2.
    # This guarantees dot <= 2 for ALL sign choices? No, if intersection <= 2, max dot is 2.
    # If intersection is 2, max dot is 2. This is valid (dist^2 = 4).
    # If we allow intersection 3, we must restrict signs.
    # The current program only picks supports with intersection <= 2 (independent set in conflict graph where edge if inter >= 3).
    # Then it picks ALL 16 signs for each selected support.
    # This is a valid construction.
    
    # Can we do better?
    # 1. Use the "inspiration" local search to find a better independent set of supports.
    # 2. After selecting supports, can we add more vectors?
    #    The current program tries to add more Type D vectors from non-selected supports.
    #    But if we have a maximal independent set, no other support is compatible (intersection >= 3).
    #    So we can't add vectors from other supports using the "all 16 signs" strategy.
    #    However, we might be able to add vectors from non-selected supports if we choose signs carefully?
    #    No, because the conflict is defined by support intersection. If inter >= 3, there exist sign choices that conflict, but also sign choices that don't.
    #    The current program's greedy fill checks each candidate vector against the current set.
    #    This is a good idea.
    
    # Let's refine the independent set search using vectorized numpy operations as suggested by the cell findings.
    
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Build conflict graph
    # Conflict if intersection size >= 3
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter_len = len(set(supports[i]) & set(supports[j]))
            if inter_len >= 3:
                conflict_supports[i, j] = True
                conflict_supports[j, i] = True
                
    # Find maximum independent set
    # Using greedy with random restarts and swap-based local search
    
    best_support_set = []
    best_size = 0
    
    rng = np.random.RandomState(42)
    
    # Precompute degree for better initial ordering
    degrees = np.sum(conflict_supports, axis=1)
    
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Greedy initialization with degree-based ordering
        order = np.arange(N_S)
        rng.shuffle(order)
        
        in_set = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        
        for v in order:
            if not blocked[v] and not in_set[v]:
                in_set[v] = True
                blocked |= conflict_supports[v]
                
        current_size = np.sum(in_set)
        
        # Swap-based local search: try to remove one node and add two non-conflicting nodes
        improved = True
        max_local_iters = 50
        local_iter = 0
        while improved and local_iter < max_local_iters:
            improved = False
            local_iter += 1
            if time.time() > deadline - 30:
                break
            
            in_set_indices = np.where(in_set)[0]
            not_in_set_indices = np.where(~in_set)[0]
            
            for v in in_set_indices:
                # If we remove v, which nodes become unblocked?
                # Nodes that were only blocked by v
                newly_unblocked = np.where(
                    (~in_set) & 
                    (~blocked | conflict_supports[v]) &  # was blocked, but v was a blocker
                    conflict_supports[v]  # conflicts with v
                )[0]
                
                # Actually, let's just try: remove v, then greedily add as many as possible
                in_set[v] = False
                blocked_after_removal = np.zeros(N_S, dtype=bool)
                for u in in_set_indices:
                    if u != v and in_set[u]:
                        blocked_after_removal |= conflict_supports[u]
                
                # Try to add nodes not in set and not blocked
                addable = np.where((~in_set) & (~blocked_after_removal))[0]
                added = 0
                for w in addable:
                    if not in_set[w] and not blocked_after_removal[w]:
                        in_set[w] = True
                        blocked_after_removal |= conflict_supports[w]
                        added += 1
                
                new_size = np.sum(in_set)
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                    break
                else:
                    # Revert
                    in_set[v] = True
                    for w in addable:
                        in_set[w] = False
                    
        if current_size > best_size:
            best_size = current_size
            best_support_set = np.where(in_set)[0].tolist()
            
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Greedy fill: try to add more Type D vectors from non-selected supports
    # This allows using supports with intersection >= 3 if signs are compatible
    
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    # Shuffle all vectors to randomize order
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)
    
    added_indices = set()
    
    # Precompute dot products with current set?
    # Current set size grows.
    # Let's check candidates one by one.
    
    for i in indices_all:
        if time.time() > deadline - 10:
            break
            
        v = all_vectors_D[i]
        
        # Check if v is already in set
        if tuple(v) in current_set_vectors_set:
            continue
            
        can_add = True
        # Check against current_set_vectors
        # Optimization: if current_set_vectors is large, this is slow.
        # But it's only ~240-300 vectors.
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
                
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))
            
    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Combine Type A and Type D
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
