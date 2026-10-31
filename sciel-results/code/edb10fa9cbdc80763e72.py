import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Generate ALL weight-3 vectors in {-1, 0, 1}^D
    # Number: C(11,3) * 2^3 = 165 * 8 = 1320
    # Norm^2 = 3.
    # Min pairwise distance^2 for distinct weight-3 vectors is 4.
    # 4 >= 3, so all are mutually compatible.
    
    candidates = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    
    # Optional: Verify constraint to be safe
    # N = 1320, matrix is 1320x1320, very fast.
    N = len(vectors)
    if N > 0:
        dots = vectors @ vectors.T
        dist2 = (vectors ** 2).sum(axis=1)[:, None] + (vectors ** 2).sum(axis=1)[None, :] - 2 * dots
        np.fill_diagonal(dist2, 0)
        min_dist2 = dist2.min()
        max_norm2 = (vectors ** 2).sum(axis=1).max()
        
        if min_dist2 < max_norm2:
            # Fallback to greedy MIS if my analysis was wrong
            conflicts = dist2 < max_norm2
            alive = np.ones(N, dtype=bool)
            selected = []
            degrees = conflicts.sum(axis=1).astype(np.int32)
            rng = np.random.default_rng(42)
            while np.any(alive) and time.time() < deadline:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                chosen = cands[rng.integers(len(cands))]
                selected.append(chosen)
                neighbors = np.where(conflicts[chosen])[0]
                to_remove = np.unique(np.concatenate(([chosen], neighbors)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            vectors = vectors[selected]
            
    return vectors
