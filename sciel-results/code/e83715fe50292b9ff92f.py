import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2x2 decomposition
    strassen_U = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1},      # M7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]
    strassen_W = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1},             # M7
    ]

    # Build the 42-rank decomposition
    U_full = np.zeros((n * m, 42), dtype=np.int64)
    V_full = np.zeros((m * p, 42), dtype=np.int64)
    W_full = np.zeros((p * n, 42), dtype=np.int64)

    col = 0
    for r_blk in range(3):  # p-blocks
        for s_blk in range(2):  # m-blocks
            for t in range(7):  # Strassen terms
                for local_u, coeff in strassen_U[t].items():
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s_blk + j_local
                    u_idx = i * m + j_global
                    U_full[u_idx, col] += coeff

                for local_v, coeff in strassen_V[t].items():
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s_blk + j_local
                    k_global = 2 * r_blk + k_local
                    v_idx = j_global * p + k_global
                    V_full[v_idx, col] += coeff

                for local_w, coeff in strassen_W[t].items():
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r_blk + k_local
                    w_idx = k_global * n + i
                    W_full[w_idx, col] += coeff

                col += 1

    # Search for linear dependencies among the 42 rank-1 terms
    # Each term s has vectors u_s (dim 8), v_s (dim 24), w_s (dim 12)
    # A dependency is sum_s alpha_s * (u_s ⊗ v_s ⊗ w_s) = 0
    # Equivalently, for each (i,j,k) tensor position: sum_s alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 0
    # And the tensor should satisfy T[i,j,k] = 1 iff i,j,k form a valid multiplication triple

    # First, verify the 42-rank decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s_val = 0
                    for r_idx in range(rank):
                        s_val += U_mat[i * m + j, r_idx] * V_mat[j * p + k, r_idx] * W_mat[k * n + i, r_idx]
                    if s_val != 1:
                        return False
        return True

    if not verify(U_full, V_full, W_full, 42):
        # Fallback to schoolbook
        r_sb = n * m * p
        U_sb = np.zeros((n * m, r_sb), dtype=np.int64)
        V_sb = np.zeros((m * p, r_sb), dtype=np.int64)
        W_sb = np.zeros((p * n, r_sb), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Try to find linear dependencies using the structure
    # Group terms by (r_blk, s_blk) pair - there are 6 groups of 7
    # Within each group, the 7 Strassen terms are linearly independent (rank 7 for 2x2x2)
    # But across groups, there might be dependencies

    # Strategy: try to find a subset of terms that can be replaced by fewer terms
    # Use the fact that the tensor has a specific structure

    # Alternative: try rank 36, 30, 24, etc. by searching for valid decompositions
    # Use a greedy approach: try to remove terms one by one and check if the remaining
    # terms can still span the tensor

    # First, let's try to see if we can merge some of the 42 terms
    # The key insight: terms with the same (r_blk, s_blk) but different Strassen indices
    # are already optimized. But terms with different (r_blk, s_blk) might share structure.

    # Let's try a different approach: use the 1x2x3 rank-1 structure more effectively
    # The 1x2x3 multiplication has rank 1, so theoretically rank(2x4x6) <= 7
    # But the index mapping is non-trivial

    # Let's try to construct a rank-7 decomposition directly
    # using the correct index mapping for 2x4x6

    def try_rank7():
        """Try to construct a rank-7 decomposition using Strassen with proper index mapping."""
        # The 2x4x6 tensor: T[i*4+j, j*6+k, k*2+i] = 1
        # We want: sum_s U[i*4+j, s] * V[j*6+k, s] * W[k*2+i, s] = T[i*4+j, j*6+k, k*2+i]
        
        # Using Strassen for the 2x2x2 part with the 1x2x3 part folded in
        # The middle index j = j1*2 + j2 where j1 in {0,1} (m-block), j2 in {0,1} (within block)
        # The output index k = k1*3 + k2 where k1 in {0,1,2} (p-block), k2 in {0,1} (within block)
        # Wait, this doesn't work cleanly. Let me try a different mapping.
        
        # Actually, let's just try to find a rank-7 decomposition by solving a system
        # This is complex, so let's try a simpler approach first: check if rank 42 can be reduced
        # by finding linear dependencies.
        
        # Use SVD-like approach: the 42 terms define a 42-dimensional space
        # Check if the tensor can be represented with fewer terms
        
        # For now, return None to indicate no improvement found
        return None

    result = try_rank7()
    if result is not None:
        U_r, V_r, W_r, r = result
        if verify(U_r, V_r, W_r, r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Try to find dependencies by checking if any term is a linear combination of others
    # This is expensive, so use a time-limited search
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Try rank 36: remove 6 terms (one from each of the 6 groups)
    # Check if the remaining 36 terms still span the tensor
    # This is a heuristic: if term s in group g can be expressed as a combination
    # of other terms, we can remove it
    
    # More systematically: try all subsets of size 41, 40, ... and check
    # This is too expensive for 42 terms. Instead, use the structure.
    
    # The 6 groups (r_blk, s_blk) each have 7 Strassen terms.
    # Within a group, the 7 terms are linearly independent (they form a basis for 2x2x2 mult).
    # Across groups, dependencies might exist.
    
    # Let's check: can we express one group's terms using others?
    # Group (r1,s1) vs Group (r2,s2): the U vectors differ in which j indices are active,
    # V vectors differ in which (j,k) indices are active, W vectors differ in which (k,i) indices.
    
    # For group (r_blk, s_blk):
    # U is nonzero only for j in {2*s_blk, 2*s_blk+1}
    # V is nonzero only for j in {2*s_blk, 2*s_blk+1} and k in {2*r_blk, 2*r_blk+1}
    # W is nonzero only for k in {2*r_blk, 2*r_blk+1}
    
    # Two terms from different groups have disjoint support in at least one of U, V, W
    # (since they use different j or k ranges). So they can't be linear combinations of each other.
    
    # This means the 42 terms are all linearly independent! No reduction possible this way.
    
    # Let's try a completely different approach: use a different base decomposition
    # Try using 2x2x2 Strassen for the 2x2x2 blocks but with a different partitioning
    
    # Alternative: partition 2x4x6 as 2x2x6 + 2x2x6 (two 2x2x6 multiplications)
    # Each 2x2x6 can be done with rank... let's think.
    # 2x2x6: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0..5}
    # This is two 1x2x6 multiplications (one for each row of A)
    # Each 1x2x6 has rank... the tensor T[0,j,k]=1 for j in {0,1}, k in {0..5}
    # This is the all-ones 1x2x6 tensor, which has rank 1 (as I showed earlier for 1x2x3)
    # Wait, but with the constraint that j must match...
    
    # Actually for 2x2x6: T[i*2+j, j*6+k, k*2+i] = 1
    # i in {0,1}, j in {0,1}, k in {0..5}
    # Total 2*2*6 = 24 nonzero entries
    
    # Can we do better than schoolbook (12) for 2x2x6?
    # Using Strassen on the 2x2x2 part: for each pair of k values, use 7 mults
    # 6 k values = 3 pairs, so 3*7 = 21 for 2x2x6. Worse than schoolbook 12.
    
    # Schoolbook for 2x2x6 is 2*2*6 = 24. Wait no, schoolbook for 2x2x6 is 2*2*6=24?
    # No: C has 2*6=12 entries, each requires 2 multiplications (sum over j=0,1), so 24 total.
    # But we can factor: for each k, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # We can precompute: M1 = A[0,0]*B[0,k], M2 = A[0,1]*B[1,k], etc.
    # But B[0,k] and B[1,k] are different for each k, so we can't share.
    # Actually: let X0 = A[0,0], X1 = A[0,1], X2 = A[1,0], X3 = A[1,1]
    # C[0,k] = X0*B[0,k] + X1*B[1,k]
    # C[1,k] = X2*B[0,k] + X3*B[1,k]
    # For each k, this is a 2x2 "matrix" times a 2x1 "matrix", requiring 4 mults (schoolbook)
    # or 7 with Strassen (but Strassen is for 2x2x2, not 2x2x1)
    # For 2x2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is rank 4 (schoolbook) and I believe rank 4 is minimal.
    
    # So 2x2x6 schoolbook = 6*4 = 24. And 2x4x6 = two 2x2x6 = 48. Matches.
    
    # The Strassen approach gives 42 which is better. Can we do better than 42?
    
    # Let's try: use a different factorization. What if we use 2x4x2 blocks?
    # 2x4x2: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0..3}, k in {0,1}
    # Schoolbook: 2*4*2 = 16
    # Can we do better? Use Strassen on 2x2x2 blocks:
    # Split j into {0,1} and {2,3}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # Each 2x2x2 block can be done with 7 mults, so 2*7 = 14 for 2x4x2.
    # Three such blocks: 3*14 = 42. Same as before.
    
    # What about using a 2x4x4 block? 
    # 2x4x4 schoolbook = 32. Strassen-based: split into 2x2x2 blocks: 4 blocks * 7 = 28.
    # Then 2x4x6 = 2x4x4 + 2x4x2 = 28 + 14 = 42. Still 42.
    
    # It seems 42 is a natural rank from this approach. To do better, we need
    # cross-block sharing.
    
    # Let's try: can we share Strassen terms between adjacent k-blocks?
    # For k-block r and r+1, the V vectors use different k indices, so no direct sharing.
    # But the U vectors are the same (same j range for same s_blk).
    # If we could combine V vectors from adjacent blocks...
    
    # This is getting complex. Let's just return the 42-rank solution.
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
