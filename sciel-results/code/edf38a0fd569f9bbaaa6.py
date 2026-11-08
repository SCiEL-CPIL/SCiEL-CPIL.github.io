"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # safety margin from 150s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Build adjacency: conflict if G[i,j] > 2
    # For weight-4 vectors, dot product is in {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    # Condition: min pairwise distance >= max norm
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # max norm = sqrt(4) = 2, so max norm^2 = 4
    # We need dist >= max_norm => dist^2 >= max_norm^2 => 8 - 2*u.v >= 4 => 2*u.v <= 4 => u.v <= 2
    # So conflict if u.v > 2, i.e., u.v >= 3
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # We want the largest independent set in this conflict graph.
    # Strategy: Multiple restarts with different greedy heuristics.
    # Heuristic 1: Minimum degree (pick vertex with fewest conflicts in remaining graph)
    # Heuristic 2: Random order (pick random vertex from remaining)
    # Heuristic 3: Weighted by some other metric
    
    best_in_set = None
    best_count = 0
    
    # Pre-compute degrees for initial state
    initial_degrees = conflict.sum(axis=1)
    
    # Number of restarts - tune based on time available
    # Each restart is O(n * k) where k is the size of the independent set
    # n = 5280, k ~ 1000-1500
    # Each step involves checking compatibility and updating remaining set
    # With numpy vectorization, each restart might take ~10-50ms
    # Budget: ~130 seconds
    # Let's try 5000 restarts with random order, plus some minimum degree ones
    
    np.random.seed(42)  # Reproducibility
    
    # Phase 1: Random restarts (fast, diverse)
    num_random_restarts = 3000
    for r in range(num_random_restarts):
        if time.time() > deadline - 30:  # Leave 30s for local search
            break
        
        perm = np.random.permutation(n)
        in_set = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        
        count = 0
        for idx in perm:
            if not remaining[idx]:
                continue
            in_set[idx] = True
            count += 1
            # Remove neighbors from remaining
            neighbors = conflict[idx]
            remaining[neighbors] = False
            remaining[idx] = False
        
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
    
    # Phase 2: Minimum degree greedy (more structured)
    if time.time() < deadline - 30:
        num_md_restarts = 200
        for r in range(num_md_restarts):
            if time.time() > deadline - 30:
                break
            
            in_set = np.zeros(n, dtype=bool)
            remaining = np.ones(n, dtype=bool)
            degrees = initial_degrees.copy().astype(np.int32)
            
            count = 0
            while remaining.any():
                # Find vertex with minimum degree among remaining
                # Add some randomness to break ties
                masked_degrees = np.where(remaining, degrees, np.inf)
                
                # Add small random perturbation to break ties randomly
                if r % 2 == 0:
                    perturbation = np.random.uniform(0, 0.5, size=n)
                    perturbation[~remaining] = np.inf
                    masked_degrees = masked_degrees + perturbation
                
                idx = np.argmin(masked_degrees)
                
                if not remaining[idx]:
                    break
                
                in_set[idx] = True
                count += 1
                remaining[idx] = False
                
                # Remove neighbors and update degrees
                neighbors = conflict[idx] & remaining
                if neighbors.any():
                    degrees[neighbors] -= 1
                    remaining[neighbors] = False
            
            if count > best_count:
                best_count = count
                best_in_set = in_set.copy()
    
    # Phase 3: Local search improvement on the best solution found
    if time.time() < deadline - 5 and best_in_set is not None:
        in_set = best_in_set.copy()
        removed = ~in_set
        
        # Try to add back removed vertices
        # Multiple passes
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            # Find removed vertices that have no conflicts with current in_set
            # conflict[v, in_set].any() tells if v conflicts with any in_set vertex
            has_conflict = conflict[:, in_set].any(axis=1)
            candidates = removed & ~has_conflict
            
            if not candidates.any():
                break
            
            # Add all candidates that don't conflict with each other
            # Use greedy approach
            cand_indices = np.where(candidates)[0]
            added_in_pass = 0
            for v_idx in cand_indices:
                if time.time() > deadline - 5:
                    break
                # Check if adding v_idx conflicts with any already in in_set
                if not conflict[v_idx, in_set].any():
                    in_set[v_idx] = True
                    removed[v_idx] = False
                    added_in_pass += 1
            
            if added_in_pass == 0:
                break
        
        best_count = in_set.sum()
        best_in_set = in_set
    
    # Phase 4: Try a different local search - swap-based
    if time.time() < deadline - 5 and best_in_set is not None:
        in_set = best_in_set.copy()
        
        # Try to improve by swapping: remove one vertex from in_set, add one from removed
        # This is more complex, so we'll do a simple version
        in_set_list = np.where(in_set)[0]
        removed_list = np.where(~in_set)[0]
        
        if len(in_set_list) > 0 and len(removed_list) > 0:
            # For each removed vertex, check if it can replace a set of in_set vertices
            # Simple version: just try adding, which we already did
            # More advanced: try 2-for-1 swaps
            # This is expensive, so we'll skip for now
            
            pass
    
    result = vectors[best_in_set]
    return result
# EVOLVE-BLOCK-END
