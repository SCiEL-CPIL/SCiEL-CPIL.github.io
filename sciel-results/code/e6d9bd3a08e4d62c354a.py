import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def local_improve(selected, vectors, dots, conflicts, deadline, rng):
    """
    Try to improve the selected set by swapping out one vector for another.
    """
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle selected indices to avoid bias
        rng.shuffle(current_selected)
        
        for idx in range(len(current_selected)):
            if time.time() > deadline:
                return current_selected
            
            s_out = current_selected[idx]
            
            # Create mask of "other selected"
            other_selected_mask = selected_mask.copy()
            other_selected_mask[s_out] = False
            
            # Find unselected vectors that do not conflict with any other selected
            if np.any(other_selected_mask):
                has_conflict = conflicts[:, other_selected_mask].any(axis=1)
            else:
                has_conflict = np.zeros(N, dtype=bool)
                
            # Candidates are those without conflict, not currently selected
            candidates = ~has_conflict & ~selected_mask
            
            # We need to pick a candidate u to replace s_out.
            if np.any(candidates):
                cand_indices = np.where(candidates)[0]
                u_in = cand_indices[rng.integers(len(cand_indices))]
                
                # Perform swap
                selected_mask[s_out] = False
                selected_mask[u_in] = True
                
                # Update list
                current_selected[idx] = u_in
                
                improved = True
                break  # Restart the loop with the new configuration
    
    return current_selected

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 150s kill
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    # Generate weight-1 vectors (all compatible with weight-4 and each other)
    weight1_vectors = generate_weight_1_vectors()
    
    if N == 0:
        return weight1_vectors
        
    # Compute dot products for weight-4 vectors
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (since norm^2 = 4, need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    conflicts = (dots > 2).astype(np.int8)
    
    # Self-conflicts don't matter since we pick distinct vectors
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS restarts
    n_restarts = 0
    mis_deadline = deadline - 15  # Leave 15s for local improvement
    while time.time() < mis_deadline:
        current_set = greedy_mis(conflicts, mis_deadline, rng)
        n_restarts += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Do local improvement on the best set found
    if len(best_set) > 0 and time.time() < deadline - 3:
        improved_set = local_improve(list(best_set), vectors, dots, conflicts, deadline - 2, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    if not best_set:
        return weight1_vectors
        
    result_vectors_w4 = vectors[best_set].astype(np.int64)
    
    # Combine weight-4 and weight-1 vectors
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors
