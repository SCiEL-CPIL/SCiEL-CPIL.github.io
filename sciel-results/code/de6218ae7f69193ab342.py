import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    D = 11
    
    # 1. Generate Type A vectors (Norm squared = 4)
    # One non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # 2. Generate Type D vectors (Norm squared = 4)
    # Four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Generate all Type D vectors
    # Key: (support_idx, sign_tuple) -> ID
    # We will store vectors in a list
    vectors_D_list = []
    key_to_id = {}
    
    for s_idx, support in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vid = len(vectors_D_list)
            vectors_D_list.append(v)
            key_to_id[(s_idx, signs)] = vid
            
    N_D = len(vectors_D_list)
    vectors_D = np.array(vectors_D_list, dtype=np.int64)
    
    # 3. Precompute Conflict Graph for Type D vectors
    # Conflict if:
    # 1. Supports intersect in exactly 3 indices.
    # 2. Signs match on those 3 indices.
    
    # Adjacency list
    adj = [set() for _ in range(N_D)]
    
    for vid, v in enumerate(vectors_D_list):
        # Get support and signs of v
        # Reconstruct support and signs from v is inefficient.
        # Better: iterate over generation structure again or store metadata.
        pass
        
    # Let's rebuild with metadata for speed
    # metadata[vid] = (support_tuple, sign_tuple)
    metadata = [None] * N_D
    vid_counter = 0
    for s_idx, support in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            metadata[vid_counter] = (support, signs)
            vid_counter += 1
            
    for vid in range(N_D):
        support_u, signs_u = metadata[vid]
        support_set_u = set(support_u)
        
        # Find all supports that intersect support_u in exactly 3
        # This means v has 3 indices from support_u and 1 index from outside.
        # Choose 3 from support_u: C(4,3) = 4
        # Choose 1 from outside (11-4=7): 7
        # Total 28 candidate supports for v.
        
        # For each candidate support, check if it exists and find matching signs.
        for i in range(4):
            # The 3 indices kept are all except support_u[i]
            kept_indices = tuple(x for x in support_u if x != support_u[i])
            # The dropped index is support_u[i]
            dropped_idx = support_u[i]
            
            # The 3 signs on kept_indices must match signs_u
            # signs_u is a tuple corresponding to support_u order.
            # We need to extract signs for kept_indices.
            # This mapping is tricky if we don't store sign per index.
            # Let's store sign per index in a dict for each vector? No, too slow.
            
            # Alternative:
            # For a candidate support S_v, which is a tuple of 4 indices.
            # We know 3 are in kept_indices.
            # The signs on those 3 must equal signs_u on those indices.
            
            # Let's just iterate over all 7 outside indices j
            for j in range(D):
                if j in support_set_u:
                    continue
                # Candidate support is kept_indices + (j,)
                # Sort it to match the canonical form used in supports list?
                # supports list contains tuples from itertools.combinations, which are sorted.
                # kept_indices is a tuple of 3 sorted ints?
                # support_u is sorted. kept_indices derived by removing one element remains sorted.
                # j is any index.
                # We need to form a sorted tuple of 4.
                cand_support = tuple(sorted(kept_indices + (j,)))
                
                if cand_support not in support_to_idx:
                    continue
                    
                s_v_idx = support_to_idx[cand_support]
                
                # Now we need to find the sign tuple for v such that:
                # 1. Support is cand_support.
                # 2. On the 3 indices in kept_indices, signs match signs_u.
                
                # signs_u is a tuple of 4 signs for support_u.
                # We need to map signs_u to the indices in kept_indices.
                # Let's create a dict for signs_u: idx -> sign
                sign_map_u = {support_u[k]: signs_u[k] for k in range(4)}
                
                # Construct the required signs for the 3 common indices
                # We need to find the position of these indices in cand_support to build the sign tuple for key_to_id.
                
                # Iterate over the 2 possible signs for the 4th index (j)
                for sign_j in [1, -1]:
                    # Build sign tuple for cand_support
                    signs_v = []
                    for idx in cand_support:
                        if idx == j:
                            signs_v.append(sign_j)
                        else:
                            # Must match sign_map_u
                            signs_v.append(sign_map_u[idx])
                            
                    signs_v_tuple = tuple(signs_v)
                    
                    key_v = (s_v_idx, signs_v_tuple)
                    if key_v in key_to_id:
                        vid_v = key_to_id[key_v]
                        if vid_v != vid:
                            adj[vid].add(vid_v)
                            adj[vid_v].add(vid)
                            
    # 4. Find Maximum Independent Set (MIS)
    # Greedy with random restarts and min-degree heuristic
    
    best_set_size = 0
    best_set = []
    
    N_restarts = 300 # Adjust based on time
    
    for r in range(N_restarts):
        if time.time() > deadline - 10:
            break
            
        available = set(range(N_D))
        current_set = []
        
        # Randomize order for tie-breaking or initial selection
        # We will use min-degree greedy
        while available:
            # Find node with min degree in available
            min_deg = float('inf')
            v_idx = -1
            
            # Optimization: If set is large, scanning all is slow.
            # But N_D is 5280.
            for v in available:
                # Degree in available subgraph
                # len(adj[v] & available) is expensive (set intersection)
                # Instead, precompute degrees? No, degrees change.
                # We can approximate or just use random if too slow.
                # Let's try a cheaper heuristic:
                # Pick random node, then check if it has low degree?
                # Or just pick random node to save time.
                # Given 5280 nodes, set intersection of size ~56 is fast.
                deg = 0
                for nb in adj[v]:
                    if nb in available:
                        deg += 1
                        if deg > min_deg:
                            break
                if deg < min_deg:
                    min_deg = deg
                    v_idx = v
            
            if v_idx == -1:
                break
                
            current_set.append(v_idx)
            available.remove(v_idx)
            
            # Remove neighbors
            for nb in adj[v_idx]:
                if nb in available:
                    available.remove(nb)
                    
        if len(current_set) > best_set_size:
            best_set_size = len(current_set)
            best_set = current_set
            # print(f"Restart {r}: Size {best_set_size}") # Debug
            
    # 5. Construct Final Vectors
    # Select the best Type D vectors
    selected_D_indices = best_set
    final_vectors_D = vectors_D[selected_D_indices]
    
    # Combine Type A and Type D
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
