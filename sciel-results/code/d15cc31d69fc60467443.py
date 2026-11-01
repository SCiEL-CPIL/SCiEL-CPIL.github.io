import numpy as np
import itertools
import time
import random

D = 11

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # 1. Generate weight-4 vectors
    # Support: 4-subset of range(11)
    # Signs: 4-tuple of +/-1
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Precompute conflicting supports for each support
    # S and S' conflict if |S ∩ S'| = 3
    # For each support S, find S' such that |S ∩ S'| = 3
    support_conflicts = [[] for _ in range(n_supports)]
    for i, s1 in enumerate(supports):
        for j, s2 in enumerate(supports):
            if i == j:
                continue
            if len(set(s1) & set(s2)) == 3:
                support_conflicts[i].append(j)
    
    # Generate all vectors
    # vector_id -> (support_idx, signs_tuple)
    vectors = []
    vec_to_support = []
    vec_to_signs = []
    
    # Also build a lookup for each support: sign_key -> [vec_indices]
    # sign_key is a 3-bit int representing signs on the 3 shared elements?
    # No, we need to know WHICH 3 elements are shared.
    # For a pair of supports (s1, s2), the shared elements are fixed.
    # We can precompute for each pair (i, j) in support_conflicts[i]:
    #   shared_elements = sorted(set(supports[i]) & set(supports[j]))
    #   s1_only = set(supports[i]) - set(shared_elements)
    #   s2_only = set(supports[j]) - set(shared_elements)
    #   The signs on shared_elements determine the conflict.
    
    # Let's structure the data:
    # For each support i:
    #   vectors_in_support[i] = list of vector indices
    #   sign_to_vecs[i] = dict mapping full sign tuple (4-tuple) to vec index?
    #   Actually, we just need to map the 3-bit key of shared signs to the 2 vecs in the other support.
    
    # Precomputation for fast neighbor lookup
    # For each support i, and each conflicting support j:
    #   shared = sorted(set(supports[i]) & set(supports[j]))
    #   # Map the positions in the sign tuple to the shared elements
    #   # This is getting complicated.
    
    # Alternative: Just build the adjacency list directly using the dense matrix method 
    # but optimized? No, the sparse method is better.
    
    # Let's stick to the dense matrix for simplicity if it's fast enough, 
    # but the current program already does that.
    # The improvement should be in the MIS solver.
    
    # Let's generate the vectors
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    N = len(all_vectors_np) # 5280
    
    # Compute dots
    dots = all_vectors_np @ all_vectors_np.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    # Convert to adjacency lists for faster traversal in greedy
    adj = [[] for _ in range(N)]
    # Let's rebuild adj using the support structure as planned.
    adj = [[] for _ in range(N)]
    
    # Map vector index to (support_idx, sign_tuple)
    v_support = []
    v_signs = []
    idx = 0
    for si, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v_support.append(si)
            v_signs.append(signs)
            idx += 1
            
    # For each support si, build a dict: key (signs on 3 specific elements) -> [vec_indices]
    # But the 3 elements depend on the partner support sj.
    # So for each pair (si, sj) in support_conflicts[si]:
    #   shared = sorted(set(supports[si]) & set(supports[sj]))
    #   # For each vector in si, extract signs on shared elements -> key
    #   # For each vector in sj, extract signs on shared elements -> key
    #   # If keys match, they conflict.
    
    # This is still complex to precompute for all pairs.
    # Given N=5280, the dense matrix approach in the current program is likely the bottleneck 
    # in terms of MIS step speed, not the matrix construction.
    # The MIS step in the current program updates degrees using `conflicts[remaining_alive][:, to_remove].sum(axis=1)`.
    # This is a matrix-vector product, which is fast in NumPy.
    # The slow part is the Python loop in `greedy_mis`.
    
    # Let's optimize the greedy MIS using the dense matrix but with better heuristics.
    # Or use the sparse adj if we can build it fast.
    
    # Let's try to build the sparse adj list efficiently.
    # Precompute for each support pair (si, sj) the mapping.
    
    # For each support si, list of vectors: [si*16, si*16+1, ..., si*16+15]
    # Signs are indexed 0..15.
    # Sign tuple for index k in 0..15:
    #   k=0: (1,1,1,1)
    #   k=1: (1,1,1,-1)
    #   ...
    #   We can precompute the 4-bit signs for each of the 16 vectors.
    
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    # sign_patterns[0] = (1,1,1,1)
    # sign_patterns[1] = (1,1,1,-1)
    # etc.
    
    # For each support si, and each conflicting sj:
    #   shared = sorted(set(supports[si]) & set(supports[sj]))
    #   # Determine which positions in the 4-tuple correspond to the shared elements.
    #   # supports[si] is a tuple of 4 positions.
    #   # shared is a tuple of 3 positions.
    #   # Map shared positions to indices in supports[si].
    #   si_pos_to_idx = {pos: i for i, pos in enumerate(supports[si])}
    #   sj_pos_to_idx = {pos: i for i, pos in enumerate(supports[sj])}
    #   shared_idx_in_si = [si_pos_to_idx[p] for p in shared]
    #   shared_idx_in_sj = [sj_pos_to_idx[p] for p in shared]
    
    #   # For each vector index k1 in 0..15 (in si):
    #   #   key1 = tuple(sign_patterns[k1][i] for i in shared_idx_in_si)
    #   #   Store k1 in dict1[key1]
    #   # Similarly for k2 in sj.
    #   # Then for each key in dict1, if key in dict2, add edges.
    
    # This is doable.
    
    adj = [[] for _ in range(N)]
    
    for si in range(n_supports):
        for sj in support_conflicts[si]:
            # Avoid double processing? 
            # support_conflicts is symmetric. 
            # We can just process all pairs.
            
            shared = sorted(set(supports[si]) & set(supports[sj]))
            
            si_pos_to_idx = {pos: i for i, pos in enumerate(supports[si])}
            sj_pos_to_idx = {pos: i for i, pos in enumerate(supports[sj])}
            
            shared_idx_in_si = [si_pos_to_idx[p] for p in shared]
            shared_idx_in_sj = [sj_pos_to_idx[p] for p in shared]
            
            # Build maps for si and sj
            map_si = {}
            for k1 in range(16):
                signs1 = sign_patterns[k1]
                key = tuple(signs1[i] for i in shared_idx_in_si)
                if key not in map_si:
                    map_si[key] = []
                map_si[key].append(si * 16 + k1)
                
            map_sj = {}
            for k2 in range(16):
                signs2 = sign_patterns[k2]
                key = tuple(signs2[i] for i in shared_idx_in_sj)
                if key not in map_sj:
                    map_sj[key] = []
                map_sj[key].append(sj * 16 + k2)
                
            # Connect
            for key, vecs1 in map_si.items():
                if key in map_sj:
                    vecs2 = map_sj[key]
                    for v1 in vecs1:
                        for v2 in vecs2:
                            adj[v1].append(v2)
                            adj[v2].append(v1)
                            
    # Now we have a sparse adjacency list.
    # Degree should be 64 for all nodes.
    
    rng = np.random.default_rng(42)
    
    best_set = []
    
    # Greedy MIS with sparse adj
    def greedy_mis_sparse(adj, deadline, rng):
        N = len(adj)
        alive = np.ones(N, dtype=bool)
        deg = np.array([len(adj[i]) for i in range(N)], dtype=np.int32)
        
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            chosen = candidates[rng.integers(len(candidates))]
            selected.append(chosen)
            
            # Remove chosen and neighbors
            to_remove = np.unique(np.concatenate(([chosen], adj[chosen])))
            alive[to_remove] = False
            
            # Update degrees
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                # For each remaining node, count how many of its neighbors were removed
                # This is O(N * deg) if done naively.
                # Better: for each removed node, decrement degree of its alive neighbors.
                for r in to_remove:
                    for nb in adj[r]:
                        if alive[nb]:
                            deg[nb] -= 1
                            
        return selected

    # Run multiple restarts
    mis_deadline = t0 + 60
    best_greedy = []
    
    while time.time() < mis_deadline:
        current_set = greedy_mis_sparse(adj, mis_deadline, rng)
        if len(current_set) > len(best_greedy):
            best_greedy = current_set
            
    best_set = best_greedy
    
    # Local improvement
    # We can use the dense conflicts matrix for local improvement checks if needed,
    # or use the adj list.
    # The current program's local_improve_add and swap are good.
    # We need to adapt them to use the adj list or keep the dense matrix.
    # Keeping the dense matrix is fine for local improvement (N=5280).
    
    # Let's use the dense matrix for local improvement as in the current program.
    # We already have `conflicts` from before? No, we skipped that.
    # Let's recompute or keep it.
    # It's better to keep it.
    
    # Re-generate dense conflicts if not present? 
    # We can just use the adj list for local improvement too.
    
    def local_improve_add_sparse(selected, adj, deadline, rng):
        N = len(adj)
        selected_mask = np.zeros(N, dtype=bool)
        for idx in selected:
            selected_mask[idx] = True
            
        current_selected = list(selected)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all unselected nodes that have no neighbors in selected
            # This is O(N * deg)
            candidates = []
            for i in range(N):
                if not selected_mask[i]:
                    # Check if any neighbor is in selected
                    conflict = False
                    for nb in adj[i]:
                        if selected_mask[nb]:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(i)
                        
            if candidates:
                for c in candidates:
                    selected_mask[c] = True
                    current_selected.append(c)
                improved = True
                
        return current_selected

    if len(best_set) > 0 and time.time() < deadline - 10:
        improved_set = local_improve_add_sparse(best_set, adj, deadline - 10, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    # Add weight-1 vectors
    weight1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            weight1_vectors.append(v)
            
    weight1_np = np.array(weight1_vectors, dtype=np.int64)
    
    result_vectors_w4 = all_vectors_np[best_set].astype(np.int64)
    
    final_vectors = np.vstack([result_vectors_w4, weight1_np])
    
    return final_vectors
