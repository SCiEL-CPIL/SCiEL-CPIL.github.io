import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    D = 11
    
    # 1. Generate weight-4 vectors (norm^2 = 4)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    w4_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(w4_vectors_np) # 5280
    
    # 2. Generate weight-1 vectors (norm^2 = 4)
    # These are vectors like [0, ..., 2, ..., 0] or [0, ..., -2, ..., 0]
    # They are compatible with each other (dot=0) and with all weight-4 vectors (dot=0)
    w1_vectors_list = []
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            w1_vectors_list.append(v)
    w1_vectors_np = np.array(w1_vectors_list, dtype=np.int8)
    n_w1 = len(w1_vectors_np) # 22
    
    # 3. Build adjacency for weight-4 vectors
    # Conflict if dot > 2
    # We use the 3-subset method for efficient sparse adjacency construction
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    adj = [[] for _ in range(n_total)]
    
    for tri, tets in tri_to_tet.items():
        if len(tets) < 2:
            continue
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                
                base1 = s1 * 16
                base2 = s2 * 16
                
                v1 = w4_vectors_np[base1:base1+16]
                v2 = w4_vectors_np[base2:base2+16]
                
                d = v1 @ v2.T
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
    
    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    # Precompute dense conflict mask for local improvement speed
    # This is 5280x5280, which is manageable
    dots = w4_vectors_np @ w4_vectors_np.T
    conflict_mask = dots > 2
    np.fill_diagonal(conflict_mask, False)
    
    rng = np.random.default_rng(42)
    
    best_k_w4 = 0
    best_selected_w4 = None
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        # Greedy Minimum Degree MIS
        while avail.any():
            if time.time() > deadline - 5:
                break
                
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            min_indices_rel = np.where(degs == min_deg)[0]
            
            # Pick one randomly among min-degree nodes
            pick_rel = min_indices_rel[rng.integers(len(min_indices_rel))]
            u = avail_indices[pick_rel]
            
            selected[u] = True
            avail[u] = False
            
            # Update neighbors
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k_w4 = np.sum(selected)
        
        if k_w4 > best_k_w4:
            best_k_w4 = k_w4
            best_selected_w4 = selected.copy()
        
        # Local Improvement
        if time.time() < deadline - 20:
            # 1. Try to add any remaining compatible vectors
            selected_indices = np.where(selected)[0]
            non_selected = np.where(~selected)[0]
            
            if len(selected_indices) > 0 and len(non_selected) > 0:
                # Check if non-selected vectors conflict with any selected
                # conflict_mask[np.ix_(non_selected, selected_indices)]
                conf_matrix = conflict_mask[np.ix_(non_selected, selected_indices)]
                can_add = ~conf_matrix.any(axis=1)
                add_indices = non_selected[can_add]
                
                for idx in add_indices:
                    selected[idx] = True
                    selected_indices = np.append(selected_indices, idx)
            
            k_w4 = np.sum(selected)
            if k_w4 > best_k_w4:
                best_k_w4 = k_w4
                best_selected_w4 = selected.copy()
                
            # 2. Swap-based local search
            # Try to swap out a selected vector for an unselected one that allows more additions?
            # Or just try to replace a selected vector with a compatible unselected one if it helps?
            # Since we already added all compatible ones, a swap is only useful if it opens up new possibilities.
            # However, a simple "shake" and re-greedy is expensive.
            # Let's try a few random swaps: remove a random selected, add a random compatible unselected.
            # This might not improve size immediately, but can change the set to allow more additions later.
            # Given time constraints, we'll stick to the "add all compatible" which is strong.
            # If we have time, we can do a few iterations of:
            # Pick random selected u, remove it. Add all compatible. If size increased, keep.
            
            if time.time() < deadline - 10:
                for _ in range(10): # Few shake iterations
                    if time.time() > deadline - 5:
                        break
                    
                    current_selected_indices = np.where(selected)[0]
                    if len(current_selected_indices) == 0:
                        break
                    
                    # Pick a random selected vector to remove
                    u_out = current_selected_indices[rng.integers(len(current_selected_indices))]
                    selected[u_out] = False
                    
                    # Recalculate compatible additions
                    new_selected_indices = np.where(selected)[0]
                    non_selected = np.where(~selected)[0]
                    
                    if len(new_selected_indices) > 0 and len(non_selected) > 0:
                        conf_matrix = conflict_mask[np.ix_(non_selected, new_selected_indices)]
                        can_add = ~conf_matrix.any(axis=1)
                        add_indices = non_selected[can_add]
                        
                        for idx in add_indices:
                            selected[idx] = True
                    
                    k_w4 = np.sum(selected)
                    if k_w4 > best_k_w4:
                        best_k_w4 = k_w4
                        best_selected_w4 = selected.copy()
                    else:
                        # If it didn't improve, we could revert, but since we're just looking for best,
                        # we can just continue. The next restart will start fresh.
                        # Actually, it's better to keep the state if it's not worse, but 
                        # since we are in a restart loop, we just update best if improved.
                        pass

    if best_selected_w4 is None:
        # Fallback: just return weight-1 vectors
        return w1_vectors_np.astype(np.int64)
    
    result_w4 = w4_vectors_np[best_selected_w4]
    
    # Combine weight-4 and weight-1 vectors
    # They are all compatible, so we can just stack them
    final_vectors = np.vstack([result_w4, w1_vectors_np])
    
    return final_vectors.astype(np.int64)
