import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate all W4 vectors
    indices = list(range(D))
    w4_vectors = []
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            w4_vectors.append(v)
    
    w4_vectors = np.array(w4_vectors, dtype=np.int8)
    n_vecs = len(w4_vectors)
    
    # 2. Compute Conflict Matrix
    dots = w4_vectors @ w4_vectors.T
    conflict_mask = (dots == 3)
    
    neighbors = []
    for i in range(n_vecs):
        nbrs = np.where(conflict_mask[i])[0]
        neighbors.append(list(nbrs))
        
    del conflict_mask
    del dots

    def greedy_construct():
        current_set = set()
        available_mask = np.ones(n_vecs, dtype=bool)
        
        # We don't need conflict_count for greedy, just available_mask.
        # When we add v, we mark all neighbors as unavailable.
        
        while True:
            # Check if any available
            # This is O(N).
            if not np.any(available_mask):
                break
                
            # Pick random available
            # To avoid O(N) scan for random, we can just pick random index and check.
            # But if available is sparse, this is slow.
            # Let's find indices of available.
            avail_indices = np.where(available_mask)[0]
            if len(avail_indices) == 0:
                break
            v = avail_indices[np.random.randint(len(avail_indices))]
            
            current_set.add(v)
            available_mask[v] = False
            
            for u in neighbors[v]:
                available_mask[u] = False
                
        return current_set

    def local_search(S):
        current_set = set(S)
        conflict_count = np.zeros(n_vecs, dtype=np.int32)
        available_mask = np.ones(n_vecs, dtype=bool)
        
        for v in current_set:
            available_mask[v] = False
            
        for v in current_set:
            for u in neighbors[v]:
                if u not in current_set:
                    conflict_count[u] += 1
                    if conflict_count[u] == 1:
                        available_mask[u] = False
                        
        best_size = len(current_set)
        
        while True:
            # Calculate gains
            gains = np.zeros(n_vecs, dtype=np.int32)
            
            for u in range(n_vecs):
                if u in current_set:
                    continue
                if conflict_count[u] == 1:
                    for v in neighbors[u]:
                        if v in current_set:
                            gains[v] += 1
                            break
                            
            if np.max(gains) <= 1: # Need gain >= 2 to increase size (remove 1, add >=2)
                # If gain is 1, size stays same.
                # If gain is 0, size decreases.
                # We only want to increase.
                break
            
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            v_to_remove = candidates[np.random.randint(len(candidates))]
            
            # Remove v_to_remove
            current_set.remove(v_to_remove)
            available_mask[v_to_remove] = True
            
            newly_available = []
            for u in neighbors[v_to_remove]:
                if u not in current_set:
                    conflict_count[u] -= 1
                    if conflict_count[u] == 0:
                        available_mask[u] = True
                        newly_available.append(u)
                        
            # Add newly available
            for u in newly_available:
                current_set.add(u)
                available_mask[u] = False
                for w in neighbors[u]:
                    if w not in current_set:
                        conflict_count[w] += 1
                        if conflict_count[w] == 1:
                            available_mask[w] = False
                            
            if len(current_set) <= best_size:
                # Revert? 
                # Since we only accept if > best_size in the main loop, 
                # and we break if gain <= 1, 
                # this branch might not be hit if we break earlier.
                # But if max_gain > 1, it's possible that adding them doesn't result in net gain?
                # No, if gain is G, we remove 1 and add G. Net G-1.
                # If G > 1, net > 0.
                # So size always increases if we take a move with gain > 1.
                pass
            else:
                best_size = len(current_set)
                
        return current_set

    best_S = set()
    best_size = 0
    
    # Initial run
    S = greedy_construct()
    S = local_search(S)
    best_S = S
    best_size = len(S)
    
    while time.time() < deadline:
        S = greedy_construct()
        S = local_search(S)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    result_vectors = w4_vectors[list(best_S)]
    return result_vectors
