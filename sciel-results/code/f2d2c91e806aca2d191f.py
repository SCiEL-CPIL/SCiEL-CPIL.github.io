"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with all weight-2 vectors (the seed's 110 vectors)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(tuple(v))
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Compute norms and check constraint
    # Max norm squared
    norms_sq = np.sum(vectors * vectors, axis=1)
    max_norm_sq = np.max(norms_sq)  # should be 2
    
    # Try to add vectors of higher weight
    # Generate candidate vectors with entries in {-1, 0, 1}
    # For efficiency, generate by weight
    
    def check_and_add(candidate, current_vectors, max_norm_sq):
        """Check if candidate can be added. Returns True if added."""
        c = np.array(candidate, dtype=np.int64)
        c_norm_sq = np.sum(c * c)
        new_max_norm_sq = max(max_norm_sq, c_norm_sq)
        
        # Check all pairwise distances
        # ||c - v||^2 = c_norm_sq + v_norm_sq - 2*c.v
        # Need >= new_max_norm_sq for all v
        diffs = current_vectors - c[None, :]
        dists_sq = np.sum(diffs * diffs, axis=1)
        if np.min(dists_sq) < new_max_norm_sq:
            return False, max_norm_sq
        
        current_vectors = np.vstack([current_vectors, c[None, :]])
        return True, new_max_norm_sq
    
    # Generate candidates by weight, from low to high
    # Weight 3 candidates
    added = True
    weight = 3
    while weight <= D and added and time.time() < deadline:
        added = False
        # Generate all weight-weight vectors with entries in {-1, 0, 1}
        # For efficiency, use combinations and signs
        if weight == 3:
            for positions in itertools.combinations(range(D), weight):
                for signs in itertools.product([-1, 1], repeat=weight):
                    if time.time() > deadline:
                        break
                    v = [0] * D
                    for pos, s in zip(positions, signs):
                        v[pos] = s
                    candidate = tuple(v)
                    # Skip if already in vectors (check by set)
                    ok, max_norm_sq = check_and_add(candidate, vectors, max_norm_sq)
                    if ok:
                        vectors = np.vstack([vectors, np.array([candidate], dtype=np.int64)])
                        added = True
        elif weight == 4:
            for positions in itertools.combinations(range(D), weight):
                for signs in itertools.product([-1, 1], repeat=weight):
                    if time.time() > deadline:
                        break
                    v = [0] * D
                    for pos, s in zip(positions, signs):
                        v[pos] = s
                    candidate = tuple(v)
                    ok, max_norm_sq = check_and_add(candidate, vectors, max_norm_sq)
                    if ok:
                        vectors = np.vstack([vectors, np.array([candidate], dtype=np.int64)])
                        added = True
        else:
            # For higher weights, this gets expensive; limit
            break
    
    return vectors
# EVOLVE-BLOCK-END
