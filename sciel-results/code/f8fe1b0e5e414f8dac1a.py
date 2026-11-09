"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s

    # Strategy: Use vectors with exactly 6 non-zero entries (norm sqrt(6)).
    # Condition for validity: Min pairwise distance >= max norm (sqrt(6)).
    # For two such vectors with supports S1, S2:
    # Dist^2 = 12 - 2 * (dot product).
    # Max dot product is intersection size j (if signs match).
    # Min Dist^2 = 12 - 2j.
    # Need 12 - 2j >= 6 => 2j <= 6 => j <= 3.
    # So supports must intersect in at most 3 indices.
    #
    # A 6-support is defined by its 5 excluded indices.
    # Let E be the excluded set. |E| = 5.
    # Intersection of supports S1, S2 <= 3 
    # <=> 11 - |E1 U E2| <= 3 
    # <=> |E1 U E2| >= 8
    # <=> 10 - |E1 n E2| >= 8
    # <=> |E1 n E2| <= 2.
    #
    # So we need a set of 5-subsets of {1..11} such that any two intersect in at most 2 elements.
    # This is a packing problem.
    
    all_indices = list(range(D))
    all_5_subsets = list(itertools.combinations(all_indices, 5))
    
    # Greedy selection of 5-subsets with pairwise intersection <= 2
    selected_excluded = []
    used_pairs = set() # Not enough, we need to check intersection size
    
    # To speed up, we can just check intersection with existing selected sets.
    # Number of candidates is C(11,5) = 462.
    # Max selected is ~16.
    # 462 * 16 * 5 operations is trivial.
    
    for subset in all_5_subsets:
        # Check if this subset intersects any existing selected subset in > 2 elements
        valid = True
        for existing in selected_excluded:
            inter_size = len(set(subset) & set(existing))
            if inter_size > 2:
                valid = False
                break
        if valid:
            selected_excluded.append(subset)
            
        # Optional: Early stop if we hit the theoretical max (16)
        if len(selected_excluded) >= 16:
            break
            
    # Generate vectors
    vectors = []
    for excluded in selected_excluded:
        support = [i for i in all_indices if i not in excluded]
        # Generate all 2^6 sign combinations
        # Indices in support are 6.
        for signs in itertools.product([1, -1], repeat=6):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        # Fallback to k=2 if something went wrong
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
                        
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
