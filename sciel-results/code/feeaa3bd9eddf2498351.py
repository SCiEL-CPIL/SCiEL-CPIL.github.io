import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Generate all integer vectors in R^11 with norm squared 4
    # Type 1: One component is +/- 2, others 0
    # Type 2: Four components are +/- 1, others 0
    
    candidates = []
    
    # Type 1: 11 positions * 2 signs = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: C(11, 4) positions * 2^4 signs = 330 * 16 = 5280 vectors
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot products
    # M[i, j] = candidates[i] . candidates[j]
    M = np.dot(candidates, candidates.T)
    
    # Conflict if dot product > 2
    # Note: M[i, i] = 4, which is > 2, so we must exclude self-conflicts
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Helper to run greedy independent set
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        # Degree of each node in the active subgraph
        # deg[i] = sum(Conflict[i, active])
        deg = Conflict[active].sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            
            # Find all active nodes with min degree
            candidates_for_pick = active_indices[active_deg == min_deg]
            
            # Pick one randomly
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors from active
            # Neighbors are where Conflict[idx] == 1
            neighbors_mask = Conflict[idx].astype(bool)
            
            # Nodes to remove: idx and its active neighbors
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            # Update degrees for remaining active nodes
            # For each remaining active node j, deg[j] -= number of removed neighbors
            removed_indices = np.where(removed_mask)[0]
            if len(removed_indices) > 0:
                # Count how many of the removed nodes were neighbors of each active node
                # This is the sum of Conflict[removed_indices, :] over the removed set
                # But we only care about active nodes
                # removal_counts[j] = sum(Conflict[removed_indices, j])
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                
                # Update deg for active nodes
                deg[active] -= removal_counts[active]
            
            # Update active mask
            active[removed_mask] = False
            
        return selected

    # Helper to run local search on a selected set
    def local_search(selected_indices, seed):
        if time.time() > deadline:
            return selected_indices
            
        rng = np.random.default_rng(seed)
        selected_set = set(selected_indices)
        selected_mask = np.zeros(N, dtype=bool)
        selected_mask[list(selected_set)] = True
        
        # We will try to improve the set
        # Loop until no improvement or timeout
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Calculate conflict counts for all nodes with the current selected set
            # conflict_counts[j] = number of selected nodes that conflict with j
            conflict_counts = Conflict @ selected_mask
            
            # Nodes not in selected set
            available = ~selected_mask
            
            # Check for nodes with 0 conflicts (can be added directly)
            zero_conflict_nodes = np.where(available & (conflict_counts == 0))[0]
            
            if len(zero_conflict_nodes) > 0:
                # Pick one randomly
                idx = rng.choice(zero_conflict_nodes)
                selected_set.add(idx)
                selected_mask[idx] = True
                improved = True
                continue
            
            # Check for nodes with exactly 1 conflict (can be added by swapping the conflicting node)
            one_conflict_nodes = np.where(available & (conflict_counts == 1))[0]
            
            if len(one_conflict_nodes) > 0:
                # To avoid getting stuck in a cycle, we can be selective.
                # But for now, just pick one randomly.
                # A better heuristic: pick the node j such that the conflicting node i has high degree,
                # or j has low degree in the remaining graph.
                # For simplicity, random pick.
                idx_j = rng.choice(one_conflict_nodes)
                
                # Find the conflicting node i
                # i is the unique selected node such that Conflict[idx_j, i] == 1
                conflicting_nodes = np.where(selected_mask & Conflict[idx_j])[0]
                if len(conflicting_nodes) == 1:
                    idx_i = conflicting_nodes[0]
                    # Perform swap: remove i, add j
                    selected_set.discard(idx_i)
                    selected_mask[idx_i] = False
                    selected_set.add(idx_j)
                    selected_mask[idx_j] = True
                    improved = True
                    continue
            
            # No direct improvement found
            break
            
        return list(selected_set)

    best_selected = []
    best_size = 0
    
    # Run multiple trials
    # Number of trials depends on time. Each greedy is fast. Local search is slower.
    # Let's do 50 greedy runs, each followed by local search.
    num_trials = 50
    
    for trial in range(num_trials):
        if time.time() > deadline - 10.0:
            break
            
        seed = 42 + trial
        # Run greedy
        sel = run_greedy(seed)
        
        # Run local search
        sel_improved = local_search(sel, seed + 1000)
        
        if len(sel_improved) > best_size:
            best_size = len(sel_improved)
            best_selected = sel_improved
            
    if not best_selected:
        # Fallback
        v = [0] * D
        v[0] = 2
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
