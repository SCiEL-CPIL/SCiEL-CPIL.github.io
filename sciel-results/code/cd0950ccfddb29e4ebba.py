import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: All vectors with exactly two non-zero entries (+/-1).
    # Norm^2 = 2. Max norm = sqrt(2).
    # Min distance between distinct such vectors is sqrt(2) (e.g., e1+e2 and e1+e3 -> diff e2-e3, norm sqrt(2)).
    # So all 220 are valid.
    candidates_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_2.append(v)
    
    baseline = np.array(candidates_2, dtype=np.int64)
    best_set = baseline
    best_size = len(baseline)
    
    # Try to find a larger set from Norm^2=4 vectors.
    # If we use only Norm^2=4, max norm = 2. Need dist^2 >= 4.
    
    candidates_4 = []
    # Type A: One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates_4.append(v)
            
    # Type B: Four entries +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)
    n4 = len(candidates_4)
    
    if time.time() > deadline - 20:
        return best_set
    
    # Compute conflict matrix for candidates_4
    # dist^2 = n_i + n_j - 2 dot
    # All norms are 4. dist^2 = 8 - 2 dot.
    # Conflict if dist^2 < 4 => 8 - 2 dot < 4 => 2 dot > 4 => dot > 2.
    # dot is integer. So conflict if dot >= 3.
    
    dots = candidates_4 @ candidates_4.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, False)
    
    # Adjacency matrix for graph operations
    adj = conflict.astype(np.int8)
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Initial Greedy Independent Set with Random Restarts to establish a good starting point
    np.random.seed(42)
    
    best_4_set = []
    
    # Number of restarts limited by time
    max_restarts = 20
    restarts_done = 0
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 30:
            break
        restarts_done += 1
        
        remaining = np.ones(n4, dtype=bool)
        # Initial degrees
        cur_degrees = degrees.copy()
        
        selected = []
        
        # Greedy: pick node with min degree, remove it and its neighbors
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Get degrees of remaining
            rem_degrees = cur_degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            # Pick one of the min-degree nodes
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and neighbors
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees
                # deg_new[r] = deg_old[r] - count(neighbors of r in actual_remove)
                deg_decrease = adj[remaining][:, actual_remove].sum(axis=1)
                cur_degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
            
            # Time check
            if time.time() > deadline - 30:
                break
                
        if len(selected) > len(best_4_set):
            best_4_set = selected
            
    if len(best_4_set) > best_size:
        best_set = candidates_4[best_4_set]
        best_size = len(best_4_set)
        
    # Simulated Annealing Refinement
    # Start from the best set found by greedy
    if best_4_set:
        # Build initial state
        # We need to track selected nodes efficiently.
        # Let's use a boolean array 'in_set' of size n4.
        in_set = np.zeros(n4, dtype=bool)
        in_set[best_4_set] = True
        
        # For fast updates, we can maintain the count of conflicts for each node.
        # conflict_count[i] = number of neighbors of i that are in the set.
        # If i is in the set, conflict_count[i] should be 0 (since we maintain independent set).
        # If i is not in the set, conflict_count[i] is the number of blockers.
        
        # Initial conflict counts
        # For each node i, count how many neighbors are in the set.
        # conflict_count = adj @ in_set
        conflict_count = (adj @ in_set.astype(np.int32)).astype(np.int32)
        
        current_size = len(best_4_set)
        best_sa_size = current_size
        best_sa_set = best_4_set.copy()
        
        # Temperature schedule
        # Start high, cool down.
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.995
        T = T_start
        
        # Max iterations based on time
        # Each iteration is O(1) or O(degree) depending on implementation.
        # Let's aim for ~100k iterations.
        max_iter = 200000
        
        rng = np.random.default_rng(42)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Choose a move type: Add, Remove, or Swap
            # Probabilities: Add 0.4, Remove 0.3, Swap 0.3
            r = rng.random()
            
            if r < 0.4:
                # Add a node
                # Candidates are nodes not in set with conflict_count == 0
                # To be efficient, we can maintain a list of available nodes.
                # But for simplicity in SA, let's just sample.
                # However, sampling from all n4 is wasteful.
                # Let's pick a random node.
                idx = rng.integers(0, n4)
                if not in_set[idx] and conflict_count[idx] == 0:
                    # Accept move
                    in_set[idx] = True
                    current_size += 1
                    
                    # Update conflict counts for neighbors
                    neighbors = np.where(adj[idx] == 1)[0]
                    conflict_count[neighbors] += 1
                    
                    if current_size > best_sa_size:
                        best_sa_size = current_size
                        best_sa_set = np.where(in_set)[0].tolist()
                else:
                    # Reject or accept based on energy (size decrease is not possible for valid add)
                    # If invalid, we don't change state.
                    pass
                    
            elif r < 0.7:
                # Remove a node
                # Pick a random node in the set
                if current_size > 0:
                    # Get indices of nodes in set
                    # This is O(n) if we scan. Let's keep a list of selected indices?
                    # Updating a list on remove is O(N).
                    # Alternatively, just scan in_set.
                    # For n4=5302, scanning is fast.
                    selected_indices = np.where(in_set)[0]
                    if len(selected_indices) > 0:
                        idx = selected_indices[rng.integers(0, len(selected_indices))]
                        
                        # Remove
                        in_set[idx] = False
                        current_size -= 1
                        
                        # Update conflict counts for neighbors
                        neighbors = np.where(adj[idx] == 1)[0]
                        conflict_count[neighbors] -= 1
                        
                        # Acceptance: Always accept removal? No, that's just shrinking.
                        # In SA, we might want to remove to enable better adds later.
                        # But since we track best, we can just do it and let adds recover.
                        # However, standard SA accepts with probability exp(-delta/T).
                        # Here delta = -1 (size decrease).
                        # prob = exp(1/T). If T is small, this is large? No, exp(positive) > 1.
                        # Wait, objective is to MAXIMIZE size.
                        # Move: size decreases by 1. Delta = -1.
                        # Accept prob = exp(-Delta/T) = exp(1/T).
                        # This is always > 1. So we should always accept?
                        # Usually, we only reject if the move is "worse" and the probability is < 1.
                        # Here, any decrease is worse.
                        # If we always accept decreases, we will shrink to 0.
                        # We need to be careful.
                        # Better strategy: Only remove if it allows a swap or if we are stuck?
                        # Or use a "swap" move which is size-neutral or size-improving.
                        
                        # Let's revert the removal if we decide to reject.
                        # But since exp(1/T) > 1, we always accept in standard formula?
                        # Actually, if we define Energy = -Size, then Delta E = +1.
                        # Accept prob = exp(-1/T).
                        # This is < 1. So we might reject.
                        
                        accept_prob = np.exp(-1.0 / T)
                        if rng.random() > accept_prob:
                            # Revert
                            in_set[idx] = True
                            current_size += 1
                            neighbors = np.where(adj[idx] == 1)[0]
                            conflict_count[neighbors] += 1
                            
            else:
                # Swap: Remove one, Add one
                # Pick a random node in set (to remove) and a random node not in set (to add)
                # This is complex to ensure validity.
                # Alternative: Pick a node in set, remove it, then try to add a neighbor?
                # Or just do two separate moves: Remove then Add.
                # Let's try a "Kempe-chain" like swap:
                # Pick a random node u in set.
                # Pick a random node v not in set.
                # If we remove u, does v become addable?
                # v is addable if conflict_count[v] - (1 if adj[u,v] else 0) == 0.
                
                if current_size > 0:
                    selected_indices = np.where(in_set)[0]
                    if len(selected_indices) > 0:
                        u = selected_indices[rng.integers(0, len(selected_indices))]
                        
                        # Try to find a v to swap with
                        # v must not be in set.
                        # v must have conflict_count[v] == 1 and adj[u,v]==1 (so removing u makes it 0)
                        # OR conflict_count[v] == 0 (but then we could just add v without removing u, which is better).
                        # So we look for v with conflict_count[v] == 1 and adj[u,v]==1.
                        
                        # Get neighbors of u
                        neighbors_u = np.where(adj[u] == 1)[0]
                        
                        # Filter neighbors that are not in set and have conflict_count 1
                        # Note: conflict_count includes u if u is in set and adj[u,v]=1.
                        # So for v in neighbors_u (and not in set), conflict_count[v] >= 1.
                        # If conflict_count[v] == 1, removing u makes it 0.
                        
                        candidates_v = []
                        for v in neighbors_u:
                            if not in_set[v] and conflict_count[v] == 1:
                                candidates_v.append(v)
                                
                        if candidates_v:
                            v = candidates_v[rng.integers(0, len(candidates_v))]
                            
                            # Perform Swap
                            # Remove u
                            in_set[u] = False
                            # Add v
                            in_set[v] = True
                            # Size remains same
                            
                            # Update conflict counts
                            # Neighbors of u: count decreases by 1
                            # Neighbors of v: count increases by 1
                            # Note: u and v are neighbors.
                            # u's count: was 0 (in set). Now not in set.
                            # v's count: was 1. Now 0 (in set).
                            
                            # For all neighbors of u:
                            neigh_u = np.where(adj[u] == 1)[0]
                            conflict_count[neigh_u] -= 1
                            
                            # For all neighbors of v:
                            neigh_v = np.where(adj[v] == 1)[0]
                            conflict_count[neigh_v] += 1
                            
                            # Special case: u is neighbor of v.
                            # u's conflict count changed?
                            # u was in set, so its conflict count was 0.
                            # Now u is not in set. Its conflict count is sum(adj[u] @ in_set).
                            # We updated neigh_u (which includes v) and neigh_v (which includes u).
                            # Let's trace u:
                            # u is in neigh_v. So conflict_count[u] += 1.
                            # u is in neigh_u? No, adj[u,u]=0.
                            # So conflict_count[u] becomes 1. (Because v is in set and adjacent).
                            # This is correct.
                            
                            # Let's trace v:
                            # v is in neigh_u. So conflict_count[v] -= 1. (Was 1, now 0).
                            # v is in neigh_v? No.
                            # So conflict_count[v] becomes 0.
                            # This is correct.
                            
                            # Other neighbors:
                            # w neighbor of u, not v: count -= 1. Correct (u left set).
                            # w neighbor of v, not u: count += 1. Correct (v entered set).
                            
                            # Since size is unchanged, we don't update best_sa_size unless we want to track sets?
                            # We only update best if size increases.
                            # So we don't need to update best_sa_set here.
                            
                            # However, this swap allows escaping local optima where no single add is possible.
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        if best_sa_size > best_size:
            best_size = best_sa_size
            best_set = candidates_4[best_sa_set]
            
    return best_set
