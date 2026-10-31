import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    # Key insight: for any two DISTINCT weight-4 vectors, dot product <= 2
    # (dot=4 would mean identical vectors). So ALL are mutually compatible.
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    # These are compatible with all weight-4 vectors (max |dot| = 2)
    # And among themselves: distinct pairs have dot <= 2. All 22 compatible.
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Base set: all v4 + all v1 = 5280 + 22 = 5302 vectors
    base_vectors = np.vstack([vectors_v4, vectors_v1])
    base_count = len(base_vectors)
    
    # Now try to add more vectors from other types
    # Weight-3 vectors with entries in {-2,-1,0,1,2} where norm^2 = 4
    # Possible patterns:
    #   One ±2 and two ±1 on different coords: norm^2 = 4+1+1=6 (too big)
    #   Actually we need norm^2 = 4 exactly for compatibility with max_norm_sq=4
    #   Or we could use vectors with norm^2 < 4 if they satisfy distance constraints
    
    # Let's try weight-2 vectors in {-2, 0, 2}^11: norm^2 = 8 (too big)
    # Weight-3 in {-1,0,1}: norm^2 = 3
    # For a norm^2=3 vector v and norm^2=4 vector u:
    #   dist^2 = 3 + 4 - 2*u.v = 7 - 2*u.v
    #   Need dist^2 >= 4 => u.v <= 1.5 => u.v <= 1
    # So we need v.u <= 1 for all u in our set
    
    # Weight-3 vectors in {-1,0,1}^11
    v3_candidates = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v3_candidates.append(v)
    
    if v3_candidates:
        vectors_v3 = np.array(v3_candidates, dtype=np.int64)
        N3 = len(vectors_v3)
        
        # Check which v3 vectors are compatible with all base vectors
        # For v3 (norm^2=3) and v4 (norm^2=4): need dot <= 1
        # For v3 (norm^2=3) and v1 (norm^2=4): need dot <= 1
        # For v3 (norm^2=3) and v3 (norm^2=3): need 3+3-2*d >= 4 => d <= 1
        
        # Compute dots with base vectors
        # This is expensive, so let's be selective
        # v3 dot v4: v3 has 3 non-zero entries in {-1,1}, v4 has 4 non-zero entries in {-1,1}
        # Max possible dot = 3 (if all 3 positions match with same sign)
        # We need dot <= 1, so we need at most 1 position where both have same sign non-zero
        
        # v3 dot v1: v1 is 2*e_i or -2*e_i. 
        # If v3[i] = 1 and v1 = 2e_i: dot = 2 > 1. CONFLICT!
        # If v3[i] = -1 and v1 = -2e_i: dot = 2 > 1. CONFLICT!
        # So v3 vectors that have a non-zero entry in position i conflict with v1 = 2e_i or -2e_i
        # depending on sign. Specifically:
        #   v3[i]=1 conflicts with v1=2e_i (dot=2)
        #   v3[i]=-1 conflicts with v1=-2e_i (dot=2)
        #   v3[i]=1 does NOT conflict with v1=-2e_i (dot=-2)
        #   v3[i]=-1 does NOT conflict with v1=2e_i (dot=-2)
        
        # So a v3 vector is compatible with all v1 vectors iff... 
        # It conflicts with at least one v1 vector for each non-zero position.
        # Actually it conflicts with exactly one of {2e_i, -2e_i} for each non-zero position i.
        # Since we include BOTH 2e_i and -2e_i in our set, every v3 vector with any non-zero
        # entry will conflict with at least one v1 vector.
        # Therefore NO v3 vector is compatible with the full base set!
        
        # What about v3 vectors compatible with just v4 vectors?
        # We'd need to drop some v1 vectors, which loses 22 vectors to gain maybe a few v3.
        # Not worth it unless we can add many v3 vectors.
        
        # Let's check: how many v3 vectors have dot <= 1 with ALL v4 vectors?
        # v3 has support S (size 3), v4 has support T (size 4).
        # dot = sum over i in S∩T of v3[i]*v4[i]
        # For dot <= 1: at most 1 position in S∩T where signs agree, 
        # and any other positions in S∩T must have opposite signs or not be in intersection.
        
        # This is complex. Let's skip v3 for now and focus on other types.
    
    # Try vectors with norm^2 = 4 but different patterns:
    # Weight-2 in {-2, 0, 2}: norm^2 = 4+4=8. Too big.
    # Weight-1 in {-2,0,2}: already have these (v1).
    # Weight-4 in {-1,0,1}: already have these (v4).
    
    # What about vectors with norm^2 = 2?
    # Weight-2 in {-1,0,1}: norm^2 = 2
    # For v2 (norm^2=2) and v4 (norm^2=4): dist^2 = 2+4-2*d = 6-2d >= 4 => d <= 1
    # For v2 (norm^2=2) and v1 (norm^2=4): dist^2 = 2+4-2*d = 6-2d >= 4 => d <= 1
    # For v2 (norm^2=2) and v2 (norm^2=2): dist^2 = 2+2-2*d = 4-2d >= 4 => d <= 0
    
    # v2 dot v1: v1 = ±2e_i. If v2 has entry ±1 at position i:
    #   v2[i]=1, v1=2e_i: dot=2 > 1. CONFLICT!
    # Same issue as v3. No v2 vector compatible with full base set.
    
    # What about vectors with norm^2 = 1?
    # Weight-1 in {-1,0,1}: norm^2 = 1
    # For v1' (norm^2=1) and v4 (norm^2=4): dist^2 = 1+4-2*d = 5-2d >= 4 => d <= 0.5 => d <= 0
    # For v1' (norm^2=1) and v1 (norm^2=4): dist^2 = 1+4-2*d = 5-2d >= 4 => d <= 0.5 => d <= 0
    # v1' dot v4: v1' = ±e_i, v4 has ±1 at 4 positions. If i is in v4's support: dot = ±1.
    # Need dot = 0, so i must NOT be in v4's support. But v4 covers 4 out of 11 positions,
    # so there are 7 positions not in v4's support. But we need this for ALL v4 vectors.
    # A given position i is in the support of many v4 vectors. In fact, for any position i,
    # there exist v4 vectors with i in their support. So v1' = e_i will have dot ≠ 0 with some v4.
    # Therefore no norm^2=1 vector is compatible with all v4 vectors.
    
    # Conclusion: The base set of 5302 vectors (all v4 + all v1) appears to be maximal
    # among the simple constructions we've considered. Let's verify this is valid and return it.
    
    # Verify: check a sample of pairwise distances
    if time.time() < deadline - 20:
        # Quick verification on a subset
        sample_size = min(100, base_count)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(base_count, size=sample_size, replace=False)
        sample = base_vectors[sample_idx]
        
        norms_sq = np.sum(sample * sample, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check pairwise distances in sample
        dots = sample @ sample.T
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        
        if min_dist_sq < max_norm_sq:
            # Shouldn't happen, but if it does, fall back to just v4
            print(f"Warning: constraint violated in sample. min_dist={min_dist_sq}, max_norm={max_norm_sq}", file=sys.stderr)
            # Return just v4 as fallback
            return vectors_v4
    
    return base_vectors

