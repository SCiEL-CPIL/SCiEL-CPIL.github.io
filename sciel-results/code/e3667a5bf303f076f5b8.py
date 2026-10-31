import numpy as np
import itertools
import time
import sys

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

def greedy_mis_spectral(conflicts, deadline, rng, use_spectral=True):
    """
    Greedy MIS with optional spectral guidance for tie-breaking.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    # Pre-compute spectral vector if requested
    spectral_scores = None
    if use_spectral and N > 100:
        try:
            # Use power iteration for the top eigenvector of the conflict matrix
            # Conflict matrix is symmetric, so eigenvectors are orthogonal.
            # We want nodes that are "peripheral", so we look at the principal eigenvector.
            # Nodes with small absolute values in the principal eigenvector are often less central.
            A = conflicts.astype(np.float32)
            # Power iteration
            v = np.random.rand(N).astype(np.float32)
            v /= np.linalg.norm(v)
            for _ in range(20): # 20 iterations is usually enough for rough guidance
                v_new = A @ v
                norm = np.linalg.norm(v_new)
                if norm < 1e-6:
                    break
                v = v_new / norm
            spectral_scores = np.abs(v)
        except Exception:
            spectral_scores = None

    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            if spectral_scores is not None:
                # Prefer candidates with lower spectral score (less central)
                # Add small random noise to break ties
                scores = spectral_scores[candidates] + rng.random(len(candidates)) * 0.1
                min_idx = np.argmin(scores)
                chosen = candidates[min_idx]
            else:
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
            # Subtract 1 for each removed neighbor
            # conflicts[remaining_alive, to_remove] is a boolean matrix
            # Summing along axis 1 gives the number of removed neighbors for each remaining node
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def saturate_set(selected_indices, conflicts, deadline):
    """
    Expand an independent set to a maximal one by adding all non-conflicting nodes.
    """
    N = len(conflicts)
    in_set = np.zeros(N, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
    
    # Initial blocked set: neighbors of all in_set nodes
    # We can compute this efficiently
    if np.any(in_set):
        blocked = conflicts[in_set].any(axis=0)
    else:
        blocked = np.zeros(N, dtype=bool)
    
    # Iterate to add any node that is not in_set and not blocked
    while True:
        if time.time() > deadline:
            break
        
        candidates = np.where(~in_set & ~blocked)[0]
        if len(candidates) == 0:
            break
        
        # Add all candidates
        in_set[candidates] = True
        # Update blocked
        # We need to OR the rows of conflicts corresponding to the new candidates
        new_blocked = conflicts[candidates].any(axis=0)
        blocked |= new_blocked
        
        # Check if we actually added anything (it should, but good to be safe)
        # If no new nodes were added, the loop would have broken on candidates==0
        
    return np.where(in_set)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Generate candidates
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    dots = vectors @ vectors.T
    # Conflict if dot > 2 (for weight 4, condition is dot <= 2)
    conflicts = dots > 2
    
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    # Weight 4 graph is smaller (5280 nodes) so we can afford more restarts
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_rng = np.random.default_rng(42 + i)
        
        # Alternate between spectral and non-spectral to diversify
        use_spec = (i % 2 == 0)
        current_set = greedy_mis_spectral(conflicts, deadline, current_rng, use_spectral=use_spec)
        
        # Saturate the set to make it maximal
        saturated_set = saturate_set(current_set, conflicts, deadline)
        
        if len(saturated_set) > best_size:
            best_size = len(saturated_set)
            best_set = saturated_set
            
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)

if __name__ == "__main__":
    construct()
