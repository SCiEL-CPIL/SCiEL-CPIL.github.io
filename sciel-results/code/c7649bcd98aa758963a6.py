import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11
    # For distinct weight-4 vectors, max dot product is 2 (they can share at most 3 positions with same sign,
    # or 2 positions with same sign and 1 with opposite, etc. Actually let's think carefully:
    # Two weight-4 vectors have supports of size 4 in 11 dimensions.
    # Intersection size can be 0, 1, 2, 3, or 4.
    # If intersection is 4 (same support), max dot = 4 if same signs, or less.
    # Wait, if they have the same support, they could be identical (dot=4) or differ in signs.
    # If they are distinct but same support: at least one sign differs, so dot <= 2 (e.g., 3 agree, 1 disagree: 3-1=2, or 2 agree 2 disagree: 0, etc.)
    # If intersection is 3: max dot = 3 (all 3 agree)
    # If intersection is 2: max dot = 2
    # If intersection is 1: max dot = 1
    # If intersection is 0: dot = 0
    # So max dot product for distinct weight-4 vectors is 3 (when they share 3 positions with same sign).
    # Wait, let me reconsider. If two vectors share 3 positions with the same sign, that's dot=3.
    # dist^2 = 4 + 4 - 2*3 = 2. But max_norm^2 = 4. So dist^2 = 2 < 4. That's a conflict!
    # So they are NOT all mutually compatible.
    
    # Let me re-examine. The current program computes a conflict graph and finds MIS.
    # The inspirations score 2.5727 which is higher. Let me think about what gives that score.
    
    # Score = k / seed. Seed score is 2.4727. If seed k is about 5280/something... 
    # Actually, the seed is likely a fixed value. Score 2.4727 with k vectors means k/seed = 2.4727.
    # If seed = 2136, then k = 2.4727 * 2136 ≈ 5282. That's close to 5280.
    # Inspiration scores 2.5727, so k = 2.5727 * 2136 ≈ 5496. That's more than 5280.
    
    # Hmm, let me think differently. Maybe the seed is different.
    # Actually, let me just focus on finding the maximum kissing number configuration.
    
    # For weight-4 vectors with norm^2 = 4:
    # Two vectors conflict if dist^2 < 4, i.e., 8 - 2*dot < 4, i.e., dot > 2.
    # So conflict when dot >= 3.
    # dot = 3 happens when they share exactly 3 positions with same sign (and the 4th positions differ).
    
    # So I need a large subset of weight-4 vectors where no two have dot >= 3.
    # This is equivalent to: for any two vectors in the subset, they share at most 2 positions with the same sign.
    
    # Actually, dot can also be negative. Let me just use the conflict criterion: dot > 2 means conflict.
    
    # Let me generate all weight-4 vectors and find a large independent set in the conflict graph
    # where conflict means dot > 2.

    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (for distinct vectors)
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Now find a large independent set. Use greedy approach with multiple restarts.
    
    def greedy_mis(conflict_matrix, n, deadline, rng_seed=42):
        alive = np.ones(n, dtype=bool)
        selected = []
        degrees = conflict_matrix.sum(axis=1).astype(np.int32)
        rng = np.random.default_rng(rng_seed)
        
        while np.any(alive) and time.time() < deadline:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflict_matrix[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)
        
        return selected
    
    best_selected = []
    restart_deadline = deadline - 5
    
    # Try multiple random restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > restart_deadline:
            break
        seed = 42 + restart
        selected = greedy_mis(conflicts, N, restart_deadline, rng_seed=seed)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    result_vectors = vectors[best_selected]
    return result_vectors
