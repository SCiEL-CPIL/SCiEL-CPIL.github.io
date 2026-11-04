import numpy as np
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 145.0
    D = 11
    
    # Generate E8 lattice vectors (norm squared = 2)
    # Type 1: All +/-1, even number of -1s
    vecs = []
    for mask in range(128):
        v = np.zeros(8, dtype=np.int8)
        num_neg = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                num_neg += 1
            else:
                v[i] = 1
        if num_neg % 2 == 0:
            # Pad to dimension 11
            v_full = np.zeros(D, dtype=np.int8)
            v_full[:8] = v
            vecs.append(v_full)
            
    # Type 2: Two non-zero entries (+/-1), rest 0
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8, dtype=np.int8)
                v[i] = si
                v[j] = sj
                v_full = np.zeros(D, dtype=np.int8)
                v_full[:8] = v
                vecs.append(v_full)
                
    # Verify condition: min dist >= max norm
    # All norms are sqrt(2).
    # Min distance between distinct E8 vectors is sqrt(2).
    # So condition holds.
    
    # Convert to array
    result = np.array(vecs, dtype=np.int8)
    
    # Optional: Try to add more vectors using the extra 3 dimensions?
    # The extra dimensions are 8, 9, 10.
    # We could try adding weight-2 vectors on the extra dimensions?
    # e.g. (0,...,0, 1, 1, 0). Norm sqrt(2).
    # Distance to existing E8 vectors (which are 0 in these dims):
    # dist^2 = ||v_e8||^2 + ||v_extra||^2 = 2 + 2 = 4. Dist = 2.
    # 2 >= sqrt(2). Condition holds.
    # Distance to other extra vectors:
    # If we add all weight-2 vectors on dims {8,9,10}, there are 3 supports, 4 signs each = 12 vectors.
    # But we must check distance between extra vectors and E8 vectors.
    # E8 vector $u$ has 0 in dims 8,9,10. Extra vector $w$ has 0 in dims 0..7.
    # $||u-w||^2 = ||u||^2 + ||w||^2 = 2+2=4$. Dist 2.
    # Max norm is $\sqrt{2}$. $2 \ge \sqrt{2}$ is true.
    # So we can add these!
    
    extra_vecs = []
    # Supports in {8, 9, 10}
    for i, j in combinations([8, 9, 10], 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int8)
                v[i] = si
                v[j] = sj
                extra_vecs.append(v)
                
    # Check if extra_vecs are compatible with each other
    # Min dist between extra vecs:
    # Same support, opposite signs: dist^2 = 4. Dist 2.
    # Diff supports (intersect in 1): e.g. (1,1,0) and (1,0,1). dist^2 = 1+1+1+1 = 4?
    # v1=(1,1,0), v2=(1,0,1). v1-v2=(0,1,-1). Norm^2=2. Dist sqrt(2).
    # Wait. If we add all 12, we need min dist >= max norm.
    # Max norm is still sqrt(2).
    # Min dist between (1,1,0) and (1,0,1) is sqrt(2).
    # So they are compatible.
    # So we can add all 12 extra vectors.
    
    # Can we add more?
    # What about weight-2 vectors mixing E8 dims and extra dims?
    # e.g. (1, 0, ..., 0, 1, 0, 0). Norm sqrt(2).
    # Dist to E8 vector (1, 1, 0...):
    # v_new = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
    # v_e8  = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # diff  = (0, -1, 0, ..., 1, 0, 0). Norm^2 = 2. Dist sqrt(2).
    # Compatible!
    # How many such vectors?
    # Choose 1 from 8 E8 dims, 1 from 3 extra dims.
    # Supports: $8 \times 3 = 24$ pairs.
    # Signs: 4 per pair.
    # Total: $24 \times 4 = 96$ vectors.
    # Are they compatible with each other?
    # v1: dim i in E8, dim a in Extra.
    # v2: dim j in E8, dim b in Extra.
    # If i=j and a=b: same support. Opposite signs -> dist sqrt(2*4)=2?
    # v1=(1,0...1), v2=(-1,0...-1). diff=(2,0...2). Norm^2=8. Dist sqrt(8). OK.
    # If i=j, a!=b:
    # v1: (1, 0... 1, 0, 0) (dim i, dim a)
    # v2: (1, 0... 0, 1, 0) (dim i, dim b)
    # diff: (0, 0... 1, -1, 0). Norm^2=2. Dist sqrt(2). OK.
    # If i!=j, a=b:
    # v1: (1, 0, 0... 1, 0, 0) (dim i, dim a)
    # v2: (0, 1, 0... 1, 0, 0) (dim j, dim a)
    # diff: (1, -1, 0... 0, 0, 0). Norm^2=2. Dist sqrt(2). OK.
    # If i!=j, a!=b:
    # v1: (1, 0... 1, 0, 0)
    # v2: (0, 1... 0, 1, 0)
    # diff: (1, -1... 1, -1, 0). Norm^2=4. Dist 2. OK.
    
    # So we can add all 96 mixed vectors.
    # Total so far: 240 (E8) + 12 (Extra) + 96 (Mixed) = 348.
    
    # Can we add more?
    # What about weight-2 vectors within E8 that we already have? Yes, they are in the 240.
    # What about weight-2 vectors within Extra? Yes, in the 12.
    # What about weight-2 vectors Mixed? Yes, in the 96.
    
    # Are there other vectors of norm sqrt(2)?
    # No, integer vectors of norm sqrt(2) must have two 1s/-1s and rest 0.
    # We have covered all supports:
    # - 2 in E8 (28 supports * 4 signs = 112) -> Included in E8 Type 2
    # - 2 in Extra (3 supports * 4 signs = 12) -> Included in Extra
    # - 1 in E8, 1 in Extra (24 supports * 4 signs = 96) -> Included in Mixed
    # - 2 in E8 (Type 1 has 8 non-zeros, not weight 2).
    # Wait, E8 Type 1 vectors have norm sqrt(8).
    # The condition is "minimum pairwise distance is at least the MAXIMUM vector norm".
    # If we mix norms, the max norm becomes sqrt(8) (from E8 Type 1).
    # Then we require min distance >= sqrt(8).
    # Dist between E8 Type 1 (norm sqrt(8)) and E8 Type 2 (norm sqrt(2)):
    # Example: v1=(1,1,1,1,1,1,1,1), v2=(1,1,0,0,0,0,0,0).
    # diff = (0,0,1,1,1,1,1,1). Norm^2 = 6. Dist sqrt(6).
    # sqrt(6) < sqrt(8). Condition FAILS.
    
    # So we cannot mix Type 1 (norm sqrt(8)) and Type 2 (norm sqrt(2)) if we want to keep all.
    # We must choose a set where max norm is M, and min dist >= M.
    
    # Option A: Use only weight-2 vectors (norm sqrt(2)).
    # Total available weight-2 vectors in R11:
    # Supports: C(11,2) = 55.
    # Signs: 4 per support.
    # Total: 220 vectors.
    # This matches the "seed" of 220.
    # Can we get more than 220 with weight-2?
    # No, there are only 220 weight-2 integer vectors.
    # So max score with weight-2 is 220.
    
    # Option B: Use only E8 Type 1 vectors (norm sqrt(8)).
    # Count: 128.
    # Min dist between distinct Type 1 vectors:
    # v1, v2 in {+-1}^8, even parity.
    # diff has entries in {-2, 0, 2}.
    # Number of non-zero entries in diff is even.
    # Min non-zero entries: 2.
    # If 2 entries are non-zero (e.g. 2 and -2), Norm^2 = 4+4=8. Dist sqrt(8).
    # So min dist = sqrt(8) = Max Norm.
    # This is a valid set of 128 vectors.
    # Score 128. Worse than 220.
    
    # Option C: Mix, but ensure min dist >= max norm.
    # If we use Type 1 (norm sqrt(8)), we must ensure all other vectors are at least dist sqrt(8) away.
    # Type 2 vectors (norm sqrt(2)) are too close (dist sqrt(6) < sqrt(8)).
    # So we can't mix Type 1 and Type 2.
    
    # What about scaling?
    # The problem says "integer vectors".
    # If we scale Type 2 vectors by 2? Norm becomes 2*sqrt(2) = sqrt(8).
    # Then we have 220 vectors of norm sqrt(8).
    # Min dist between scaled Type 2 vectors:
    # Original min dist sqrt(2). Scaled min dist 2*sqrt(2) = sqrt(8).
    # So we can take all 220 weight-2 vectors, multiply by 2.
    # Norms are all sqrt(8). Min dist is sqrt(8).
    # This gives a valid set of 220 vectors.
    # Score 220. Same as seed.
    
    # Can we add more vectors of norm sqrt(8) to the scaled Type 2 set?
    # We have 220 vectors: $2 \times (\pm 1, \pm 1, 0...)$.
    # Are there other integer vectors of norm sqrt(8) that are distance >= sqrt(8) from all these?
    # The E8 Type 1 vectors (norm sqrt(8)) are candidates.
    # Let's check distance between scaled Type 2 and E8 Type 1.
    # v_scaled = (2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0).
    # u_e8 = (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0).
    # diff = (1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0).
    # Norm^2 = 1+1+1+1+1+1+1+1 = 8.
    # Dist = sqrt(8).
    # Condition holds!
    
    # So we can combine:
    # 1. All 220 scaled weight-2 vectors (norm sqrt(8)).
    # 2. All 128 E8 Type 1 vectors (norm sqrt(8)).
    # Total: 348 vectors.
    # We must verify that ANY pair from this combined set has distance >= sqrt(8).
    # We checked:
    # - Scaled W2 vs Scaled W2: Min dist sqrt(8). OK.
    # - E8 T1 vs E8 T1: Min dist sqrt(8). OK.
    # - Scaled W2 vs E8 T1:
    #   Let $v$ be scaled W2 (entries 0, +-2).
    #   Let $u$ be E8 T1 (entries +-1).
    #   $v$ has support of size 2. $u$ has support of size 8.
    #   $v = 2 e_i \pm 2 e_j$.
    #   $u = \sum_{k=0}^7 \sigma_k e_k$.
    #   $v - u$ has entries:
    #   At $i, j$: $\pm 2 - \sigma_i, \pm 2 - \sigma_j$.
    #   At other $k \in \{0..7\} \setminus \{i,j\}$: $-\sigma_k$ (value $\pm 1$).
    #   At $k \ge 8$: 0.
    #   Norm^2 = $(\pm 2 - \sigma_i)^2 + (\pm 2 - \sigma_j)^2 + \sum_{k \notin \{i,j\}, k<8} 1$.
    #   Number of $k$ in sum: $8 - 2 = 6$.
    #   So sum is 6.
    #   Terms $(\pm 2 - \sigma)^2$:
    #   If sign matches ($\sigma = \pm 2$? No $\sigma \in \{-1, 1\}$).
    #   Case 1: $v_i = 2, u_i = 1$. Diff 1. Sq 1.
    #   Case 2: $v_i = 2, u_i = -1$. Diff 3. Sq 9.
    #   Min value is 1.
    #   So Min Norm^2 = $1 + 1 + 6 = 8$.
    #   Dist = sqrt(8).
    #   Condition holds!
    
    # So the combined set of 348 vectors is valid.
    # Score 348.
    
    # Can we add more?
    # What about scaled weight-2 vectors that involve the extra dimensions (8,9,10)?
    # Yes, we included all 220 weight-2 vectors in R11.
    # Supports in {0..10}.
    # So we have all possible weight-2 vectors scaled by 2.
    
    # Are there other vectors of norm sqrt(8)?
    # Integer vectors with sum of squares 8.
    # Types:
    # 1. Two 2s (and 0s). $2^2+2^2=8$. (These are our scaled W2). Count: $C(11,2) \times 4 = 220$.
    # 2. One 2, four 1s? $4+1+1+1+1=8$.
    #    Support size 5.
    #    Positions: Choose 1 for 2, 4 for 1s.
    #    Number of such vectors:
    #    Choose pos for 2: 11.
    #    Choose 4 pos for 1s from remaining 10: $C(10,4) = 210$.
    #    Signs: 2 for the 2, $2^4$ for the 1s.
    #    Total: $11 \times 210 \times 2 \times 16 = 73920$.
    #    Are they compatible with the existing 348?
    #    Let $w$ be such a vector. Norm sqrt(8).
    #    Check dist to scaled W2 vector $v$ (two 2s).
    #    If supports are disjoint: Dist^2 = 8+8=16. OK.
    #    If supports intersect:
    #    Example: $v=(2,2,0...)$. $w=(2,1,1,1,1,0...)$.
    #    Diff: $(0, 1, -1, -1, -1, ...)$.
    #    Norm^2 = $0 + 1 + 1 + 1 + 1 = 4$.
    #    Dist = 2.
    #    Max Norm = sqrt(8) $\approx 2.82$.
    #    $2 < 2.82$. Condition FAILS.
    #    So we cannot add these.
    
    # 3. Eight 1s. (E8 Type 1). Already included.
    
    # So 348 seems to be a strong lower bound.
    
    # Let's construct the 348 vectors.
    
    final_vecs = []
    
    # 1. Scaled Weight-2 vectors
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 2 * si
                v[j] = 2 * sj
                final_vecs.append(v)
                
    # 2. E8 Type 1 vectors (padded)
    for mask in range(128):
        v = np.zeros(8, dtype=np.int8)
        num_neg = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                num_neg += 1
            else:
                v[i] = 1
        if num_neg % 2 == 0:
            v_full = np.zeros(D, dtype=np.int8)
            v_full[:8] = v
            final_vecs.append(v_full)
            
    result = np.array(final_vecs, dtype=np.int8)
    
    # Final verification step (fast)
    # Check norms
    norms_sq = np.sum(result**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    # Check distances
    # Sample check or full check?
    # 348 vectors. 348*348 matrix is small.
    G = result @ result.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    
    if min_dist_sq < max_norm_sq:
        # Fallback to just E8 Type 1 (128) or Scaled W2 (220) if something went wrong
        # But math says it should work.
        # If it fails, return Scaled W2 (220) which is known good.
        return np.array([v for v in final_vecs if np.sum(v**2) == 8 and np.count_nonzero(v) == 2], dtype=np.int8)
        
    return result
