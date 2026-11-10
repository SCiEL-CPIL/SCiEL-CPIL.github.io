# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Uses equinorm vectors (norm^2 = 8) from D_11 lattice roots and E_8 type roots.
    The constraint simplifies to: for all pairs u,v, (u-v)^2 >= 8, i.e., u.v <= 4.
    Since ||u||^2 = ||v||^2 = 8, we need (u-v)^2 = 16 - 2*u.v >= 8, so u.v <= 4.
    Actually, the condition is min_dist_sq >= max_norm_sq.
    If all vectors have norm^2 = 8, then max_norm_sq = 8.
    We need min over pairs of (u-v)^2 >= 8.
    (u-v)^2 = ||u||^2 + ||v||^2 - 2*u.v = 16 - 2*u.v.
    So we need 16 - 2*u.v >= 8, i.e., u.v <= 4.
    
    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    d = 11
    target_norm_sq = 8
    
    # Generate candidate vectors with norm^2 = 8
    # Type 1: D_11 roots - permutations of (±2, ±2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # There are C(11,2) * 2^2 = 55 * 4 = 220 such vectors
    type1_vectors = []
    for i in range(d):
        for j in range(i+1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    type1_vectors.append(v)
    
    # Type 2: E_8 type roots - permutations of (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1, 0, 0, 0)
    # with an even number of minus signs
    # Number of such vectors: C(11,8) * 2^8 / 2 = 165 * 128 / 2 = 10560
    # That's a lot, but let's generate them all
    type2_vectors = []
    for positions in itertools.combinations(range(d), 8):
        # Generate all sign combinations with even number of minus signs
        remaining = [i for i in range(d) if i not in positions]
        for signs in itertools.product([-1, 1], repeat=8):
            if signs.count(-1) % 2 == 0:  # Even number of minus signs
                v = np.zeros(d, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                type2_vectors.append(v)
    
    # Combine all candidates
    all_candidates = np.array(type1_vectors + type2_vectors, dtype=np.int64)
    
    # Verify all have norm^2 = 8
    norms_sq = np.sum(all_candidates * all_candidates, axis=1)
    valid_mask = (norms_sq == target_norm_sq)
    all_candidates = all_candidates[valid_mask]
    
    # Now we need to select a subset such that for all pairs u,v in the subset: u.v <= 4
    # This is a maximum independent set problem in a graph where edges connect pairs with u.v > 4
    # Or equivalently, we want the largest set where all pairwise dot products are <= 4
    
    n_candidates = len(all_candidates)
    print(f"Total candidates: {n_candidates}")
    
    # Compute all pairwise dot products
    # This is O(n^2 * d) which for n=10780 is about 10780^2 * 11 ≈ 1.28 billion operations
    # That might be too slow in Python. Let's use NumPy vectorization.
    
    # Actually, let's use a greedy approach with vectorized distance checks
    selected_indices = []
    selected_vectors = []
    
    # For efficiency, let's work with the matrix directly
    # We'll use a greedy algorithm: repeatedly pick the candidate that has the most "room"
    
    # First, let's compute the full dot product matrix
    # This is n_candidates x n_candidates, which for 10780 is about 116 million entries
    # That's about 928 MB for int64, which is large but manageable
    
    # Actually, let's be smarter. Let's use a greedy approach with vectorized checks.
    
    remaining = list(range(n_candidates))
    best_set_size = 0
    best_selected = None
    
    # Try multiple random orderings
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Randomly shuffle candidates
        order = rng.permutation(n_candidates)
        
        selected = []
        selected_idx = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            v = all_candidates[idx]
            
            if not selected:
                selected.append(idx)
                continue
            
            # Check if v can be added: v.v' <= 4 for all selected v'
            # Vectorized check
            selected_vectors_arr = all_candidates[selected]
            dots = selected_vectors_arr @ v
            
            if np.all(dots <= 4):
                selected.append(idx)
        
        if len(selected) > best_set_size:
            best_set_size = len(selected)
            best_selected = selected.copy()
            print(f"Restart {restart}: found set of size {best_set_size}")
    
    if best_selected is None or len(best_selected) < 2:
        # Fallback
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    else:
        points = all_candidates[best_selected]
    
    return points

# EVOLVE-BLOCK-END
