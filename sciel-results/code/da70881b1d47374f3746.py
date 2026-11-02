"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    # 1. Generate candidate pool
    # We use weight 1, 2, 3, 4.
    # Weight 1: 22 vectors (norm^2 = 1)
    # Weight 2: 132 vectors (norm^2 = 2)
    # Weight 3: 352 vectors (norm^2 = 3)
    # Weight 4: 330 vectors (norm^2 = 4)
    # Total ~836 candidates.
    
    candidates = []
    for w in [1, 2, 3, 4]:
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # 2. Max-Min Diversity Heuristic
    # This is a deterministic algorithm that picks the point that is "farthest" from the current set.
    # "Farthest" is defined in terms of the constraint: we want to maximize the minimum "slack"
    # where slack_i = min_{j in selected} (dist(i, j)^2 - max(norm(i)^2, norm(j)^2)).
    # If we can pick a point with slack >= 0, it's valid.
    # We prioritize points with the largest minimum slack.
    
    # To speed this up, we maintain an array `min_slack` for each unselected candidate.
    # Initially, min_slack is infinity (or a large number).
    # When we select a point `k`, we update `min_slack` for all unselected `i`:
    # new_slack = dist(i, k)^2 - max(norm(i)^2, norm(k)^2)
    # min_slack[i] = min(min_slack[i], new_slack)
    # Then we pick the unselected `i` with the largest `min_slack[i]`.
    # If that max value is < 0, we stop (or we could try to restart with a different starting point, 
    # but Max-Min is usually robust).
    
    # However, simple Max-Min might get stuck in a local optimum depending on the start.
    # We can try multiple random starting points.
    
    best_vectors = []
    best_count = 0
    rng = np.random.default_rng(42)
    
    # We will run a few iterations of the Max-Min heuristic with different starting points.
    # Since the candidate set is small (~800), this is very fast.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random start
        start_idx = rng.integers(0, n_cand)
        
        selected_indices = [start_idx]
        selected_vectors = [candidates[start_idx]]
        
        # min_slack[i] represents the minimum slack achieved against all selected points so far.
        # Initialize to a large number.
        min_slack = np.full(n_cand, np.inf)
        min_slack[start_idx] = -np.inf # Mark as selected/invalid for selection
        
        # Pre-compute pairwise squared distances? 
        # dist_sq[i, j] = norm(i)^2 + norm(j)^2 - 2 dot(i, j)
        # We can compute this on the fly or precompute the Gram matrix.
        # Gram matrix G = candidates @ candidates.T
        # dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
        # This is 836x836, which is small.
        
        # Let's precompute the full distance matrix and the "max norm" matrix to make updates O(N) instead of O(N^2) per step?
        # Actually, updating min_slack requires:
        # slack[i, k] = dist_sq[i, k] - max(norms_sq[i], norms_sq[k])
        # We can precompute a matrix `constraint_matrix` where M[i, k] = dist_sq[i, k] - max(norms_sq[i], norms_sq[k]).
        # Then min_slack[i] = min_{k in selected} M[i, k].
        
        # Precomputation cost: O(N^2) ~ 700k ops. Negligible.
        if restart == 0:
            G = candidates @ candidates.T
            dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
            max_norms_mat = np.maximum(norms_sq[:, None], norms_sq[None, :])
            constraint_matrix = dist_sq - max_norms_mat
        
        # Main loop
        while True:
            # Find the unselected index with the maximum min_slack
            # We only care about unselected indices.
            # min_slack[selected] is -inf, so they won't be picked unless all are -inf.
            
            # Optimization: Instead of scanning all N every time, we could use a heap, 
            # but N is small enough that np.argmax is fine.
            
            next_idx = np.argmax(min_slack)
            max_slack = min_slack[next_idx]
            
            if max_slack < 0:
                # No more valid points can be added
                break
            
            # Select next_idx
            selected_indices.append(next_idx)
            selected_vectors.append(candidates[next_idx])
            min_slack[next_idx] = -np.inf
            
            # Update min_slack for all other points
            # min_slack = np.minimum(min_slack, constraint_matrix[:, next_idx])
            # But we must not update the selected point itself (already -inf).
            # And we should probably not update points that are already -inf? 
            # np.minimum handles this fine.
            
            # Vectorized update
            # constraint_matrix[:, next_idx] is a vector of size N.
            # min_slack is a vector of size N.
            # We want min_slack[i] = min(min_slack[i], constraint_matrix[i, next_idx])
            
            # Note: constraint_matrix[i, i] is 0 (dist 0, max norm norm). 
            # Wait, if i is selected, min_slack[i] is -inf.
            # If i is not selected, we update it.
            
            # One subtle point: constraint_matrix[i, next_idx] might be negative for some i.
            # That's fine, min_slack will become negative, and if it's the max, we stop.
            
            min_slack = np.minimum(min_slack, constraint_matrix[:, next_idx])
            
            # If we've selected all, break
            if len(selected_indices) == n_cand:
                break
        
        if len(selected_vectors) > best_count:
            best_count = len(selected_vectors)
            # Store a copy
            best_vectors = selected_vectors.copy()
            
            # Early stop if we reach a very high number?
            # Theoretical kissing number for D=11 is not known exactly but likely > 500.
            # Max possible from this candidate set is 836.
            # If we get close to 800, it's likely optimal for this set.
            if best_count > 800:
                break

    if not best_vectors:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(best_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
