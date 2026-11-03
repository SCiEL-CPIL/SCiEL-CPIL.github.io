import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    D = 11
    
    # 1. Generate Candidate Pool: Weight-4 vectors
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Conflict Matrix
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    # Conflict if dot > 2
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute neighbor lists for faster local search
    # neighbors[i] = list of indices j such that conflict[i, j] == 1
    # We also need compatible neighbors: compat_neighbors[i] = list of j such that conflict[i, j] == 0
    compat_neighbors = [np.where(conflict[i] == 0)[0] for i in range(n)]
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    rng = np.random.default_rng(42)
    
    def greedy_construct(alpha=0.0):
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # GRASP: Select from RCL
            # RCL = {i | remaining[i] > 0 and deg[i] <= min_deg + alpha * (max_deg - min_deg)}
            # We only consider remaining nodes
            valid_deg = deg.copy()
            valid_deg[remaining == 0] = np.inf
            max_deg = valid_deg[remaining > 0].max() if np.any(remaining > 0) else 0
            
            threshold = min_deg + alpha * (max_deg - min_deg)
            
            candidates_to_pick = np.where((deg <= threshold) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
                
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected

    def local_search(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        # incompat[i] = number of selected vectors j such that conflict[i, j] == 1
        incompat = np.zeros(n, dtype=np.int32)
        for v in sel_indices:
            # v conflicts with neighbors[v]
            # We need to increment incompat for all neighbors of v
            # neighbors are where conflict[v] == 1
            incompat[conflict[v] == 1] += 1
        
        improved = True
        max_moves = 1000
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[idx]
                
                # Candidates u such that u is compatible with S \ {v}
                # u is compatible with S \ {v} if:
                # 1. incompat[u] == 0 (compatible with all S)
                # 2. incompat[u] == 1 and conflict[u, v] == 1 (incompatible only with v)
                
                mask_in_S = selected == 1
                
                cond_0 = (incompat == 0)
                cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                Cand_v = (cond_0 | cond_1) & (~mask_in_S)
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Find a compatible pair (u, w) in cand_indices
                # Use fast randomized check
                found_pair = False
                u = -1
                w = -1
                
                # Try a few random u's
                for _ in range(20):
                    u = cand_indices[rng.integers(len(cand_indices))]
                    # Get compatible neighbors of u
                    compat_with_u = compat_neighbors[u]
                    # We need w in cand_indices AND w in compat_with_u
                    # Use isin
                    mask = np.isin(cand_indices, compat_with_u)
                    candidates_w = cand_indices[mask]
                    
                    # Remove u from candidates_w
                    # Since u is in cand_indices and u is compatible with u, u is in candidates_w
                    if len(candidates_w) > 1:
                        # Pick a random w from candidates_w
                        w = candidates_w[rng.integers(len(candidates_w))]
                        if w != u:
                            found_pair = True
                            break
                        elif len(candidates_w) == 2:
                            # The other one must be compatible
                            # If candidates_w is [u, x], and w was u, then x is the other
                            # We can just pick the other one
                            # Actually, if w==u, we can just pick the other element
                            # But we picked w randomly. If w==u, we can check if there's another.
                            # Since len > 1, there is at least one other.
                            # Let's just pick the first one that is not u
                            for cw in candidates_w:
                                if cw != u:
                                    w = cw
                                    found_pair = True
                                    break
                    elif len(candidates_w) == 1 and candidates_w[0] != u:
                        # This case shouldn't happen if u is in candidates_w
                        # But if u is not in candidates_w? 
                        # Wait, u is in cand_indices. Is u in compat_with_u? Yes.
                        # So u is always in candidates_w.
                        # So len(candidates_w) >= 1.
                        # If len == 1, it must be u. So no other w.
                        pass

                if found_pair:
                    # Perform move: remove v, add u, w
                    # Update incompat
                    # Remove v: decrement incompat for neighbors of v
                    incompat[conflict[v] == 1] -= 1
                    
                    # Add u: increment incompat for neighbors of u
                    incompat[conflict[u] == 1] += 1
                    
                    # Add w: increment incompat for neighbors of w
                    incompat[conflict[w] == 1] += 1
                    
                    selected[v] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    moves_made += 1
                    break # Restart while loop with new S
            
            # If no 1-for-2 found, try 1-for-1 swaps to escape local optima
            if not improved:
                if k_sel > 0:
                    v = sel_indices[rng.integers(k_sel)]
                    
                    mask_in_S = selected == 1
                    cond_0 = (incompat == 0)
                    cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                    Cand_v = (cond_0 | cond_1) & (~mask_in_S)
                    
                    cand_indices = np.where(Cand_v > 0)[0]
                    
                    if len(cand_indices) > 0:
                        u = cand_indices[rng.integers(len(cand_indices))]
                        
                        # Swap v with u
                        incompat[conflict[v] == 1] -= 1
                        incompat[conflict[u] == 1] += 1
                        
                        selected[v] = 0
                        selected[u] = 1
                        improved = True
                        moves_made += 1
                        break
        
        return selected, moves_made

    # Multi-Start with varying alpha
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    num_restarts = 1000 # Upper bound, limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Cycle through alphas
        alpha = alphas[restart % len(alphas)]
        
        # Phase 1: Greedy
        selected = greedy_construct(alpha)
        
        # Phase 2: Local Search
        selected, moves = local_search(selected)
        
        current_count = selected.sum()
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
    # 4. Extract Result
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
