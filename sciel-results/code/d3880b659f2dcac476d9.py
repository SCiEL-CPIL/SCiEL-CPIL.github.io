import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    def generate_candidates(weight):
        """Generate all integer vectors of given Hamming weight with entries in {-1, 1}."""
        cands = []
        for idxs in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for i, s in zip(idxs, signs):
                    v[i] = s
                cands.append(v)
        return np.array(cands, dtype=np.int8)

    def grasp_max_independent_set(candidates, max_dot, iterations=100):
        """
        Find a large subset of candidates such that pairwise dot product <= max_dot.
        Uses GRASP (Greedy Randomized Adaptive Search Procedure) with 1-for-1 (reconstruction) local search.
        """
        n = len(candidates)
        if n == 0:
            return np.array([], dtype=np.int64)
        
        # Compute Gram matrix
        G = candidates @ candidates.T
        
        # Conflict matrix: True if dot > max_dot
        conflict = (G > max_dot)
        # Ensure symmetry and no self-conflict (though dot of v with v is weight > max_dot usually? 
        # For weight 4, dot is 4. max_dot is 2. So diagonal is True. 
        # We should ignore self-conflict.
        np.fill_diagonal(conflict, False)
        
        best_size = 0
        best_mask = np.zeros(n, dtype=bool)
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # 1. Construction Phase
            selected = np.zeros(n, dtype=bool)
            available = np.ones(n, dtype=bool)
            
            while np.any(available):
                # Calculate "degree" of available nodes within the available set
                # degree[i] = number of conflicts with other available nodes
                # conflict[available, :] is a matrix. We want sum over available columns.
                # Using matrix multiplication or sum.
                # deg = (conflict[available, :] * available).sum(axis=1)
                # This is efficient enough for N=5280.
                
                avail_idx = np.where(available)[0]
                if len(avail_idx) == 0:
                    break
                
                # Compute degrees for available nodes
                # conflict_sub = conflict[np.ix_(avail_idx, avail_idx)]
                # degs = conflict_sub.sum(axis=1)
                
                # Vectorized approach:
                # degs = (conflict[:, avail_idx].T * available[avail_idx]).sum(axis=1) 
                # Actually simpler:
                degs = np.sum(conflict[:, avail_idx], axis=1)
                # We only care about degs for avail_idx
                degs_avail = degs[avail_idx]
                
                # RCL: Select nodes with lowest degree (greedy heuristic for MIS)
                # Sort by degree
                sorted_degs, sorted_indices = np.sort(degs_avail), np.argsort(degs_avail)
                
                # RCL size: e.g., 10% of available
                rcl_size = max(1, int(0.1 * len(avail_idx)))
                rcl_indices = avail_idx[sorted_indices[:rcl_size]]
                
                # Pick randomly from RCL
                chosen = rcl_indices[np.random.randint(len(rcl_indices))]
                
                selected[chosen] = True
                # Remove chosen and its neighbors from available
                # neighbors = conflict[chosen, :]
                # available[neighbors] = False
                # available[chosen] = False
                
                # Vectorized update
                available[conflict[chosen, :]] = False
                available[chosen] = False
            
            current_size = np.sum(selected)
            
            # 2. Local Search Phase (1-for-1 reconstruction)
            # Try to improve by removing one node and greedily adding back all possible nodes
            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 5:
                    break
                
                # Get indices of selected nodes
                sel_idx = np.where(selected)[0]
                
                for u in sel_idx:
                    if time.time() > deadline - 5:
                        break
                        
                    # Temporarily remove u
                    temp_selected = selected.copy()
                    temp_selected[u] = False
                    
                    # Find all nodes that can be added to temp_selected
                    # A node j can be added if it is not in temp_selected AND it doesn't conflict with any node in temp_selected
                    # conflict[:, temp_selected] is a boolean matrix (N x K)
                    # .any(axis=1) gives True for rows that have at least one conflict with temp_selected
                    has_conflict = np.any(conflict[:, temp_selected], axis=1)
                    
                    # Nodes that are not selected and have no conflict
                    can_add = (~has_conflict) & (~temp_selected)
                    
                    # New selected set
                    new_selected = temp_selected | can_add
                    new_size = np.sum(new_selected)
                    
                    if new_size > current_size:
                        selected = new_selected
                        current_size = new_size
                        improved = True
                        break # Restart local search with new solution
                
            if current_size > best_size:
                best_size = current_size
                best_mask = selected.copy()
                
        return candidates[best_mask]

    best_vectors = []
    best_score = 0

    # Try Weight 4 with GRASP
    if time.time() < deadline - 10:
        cands_4 = generate_candidates(4)
        res_4 = grasp_max_independent_set(cands_4, 2, iterations=50)
        if len(res_4) > best_score:
            best_score = len(res_4)
            best_vectors = res_4

    # Try Weight 3 with GRASP (smaller space, can do more iterations)
    if time.time() < deadline - 5:
        cands_3 = generate_candidates(3)
        # Weight 3 norm sqrt(3). Dist^2 = 3+3-2dot = 6-2dot.
        # Need Dist >= sqrt(3) (max norm). Dist^2 >= 3.
        # 6-2dot >= 3 => 2dot <= 3 => dot <= 1.5 => dot <= 1.
        res_3 = grasp_max_independent_set(cands_3, 1, iterations=100)
        if len(res_3) > best_score:
            best_score = len(res_3)
            best_vectors = res_3

    if len(best_vectors) == 0:
        vecs = []
        for i in range(D):
            v = [0]*D
            v[i] = 1
            vecs.append(v)
            v[i] = -1
            vecs.append(v)
        best_vectors = np.array(vecs, dtype=np.int64)

    return np.array(best_vectors, dtype=np.int64)
