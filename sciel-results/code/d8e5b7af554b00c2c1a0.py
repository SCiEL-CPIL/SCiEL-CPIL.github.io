"""Matrix multiplication 3,3,8 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    def _is_halfint(x):
        x2 = int(round(x * 2))
        return abs(x * 2 - x2) < 1e-8
    
    def _verify(U, V, W):
        """Verify T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] matches multiplication tensor."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_orig = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_orig[i*m+j, j*p+k, k*n+i] = 1.0
        if not np.allclose(T_recon, T_orig, atol=1e-6):
            return False
        # Check half-integer entries
        for mat in [U, V, W]:
            for val in mat.flat:
                if not _is_halfint(val):
                    return False
        return True
    
    # Strategy: Use Strassen's rank-7 for 2x2x2, embedded in 3x3x3,
    # then handle the 8 columns.
    # 
    # Key insight: 3x3x8 can be decomposed as follows.
    # The multiplication C[i,k] = sum_j A[i,j]*B[j,k] for i,k in range, j in range.
    # 
    # For 3x3x3, use the known rank-23 decomposition.
    # For the extra 5 columns (to get to 8), each adds at most 9 terms,
    # but we can be smarter.
    #
    # Actually, let's use a different approach:
    # View 3x3x8 as combining 3x3x4 + 3x3x4.
    # 3x3x4 rank: at most 24, but can we do better?
    # 
    # Let me construct 3x3x4 using 4x4 rank-9 Strassen + 1 correction.
    # No, 4x4 rank 9 is for 4x4x4.
    #
    # Simpler: use the direct construction.
    # For 3x3xn, the rank is at most 7*n for n>=2 using Strassen-like.
    # But we want minimal.
    #
    # Let me try: construct 3x3x8 with rank 48 first (verified), then optimize.
    
    # Rank-48 construction: split 8 columns into two groups of 4.
    # For each 3x3x4 block, use schoolbook rank 12. Total 24.
    # Wait, 3*3*4 = 36 schoolbook, not 12.
    # 
    # Schoolbook for 3x3x4: 3*3*4 = 36 terms. Two blocks: 72. Same as before.
    #
    # Let me use Strassen for 2x2x2 blocks within 3x3.
    # 3x3x3: partition into 2x2 block (rank 7) + remaining terms.
    # 
    # Better approach: Use the known rank-23 for 3x3x3 (from literature).
    # Then for columns 4-8, add rank 9 each = 45. Total 23+45=68. Not better.
    #
    # Let me try a completely different structural approach.
    # 
    # The tensor T[i,j,k] where U-index=(i,j), V-index=(j,k), W-index=(k,i).
    # Notice: for fixed j, we have a 3x8x3 sub-structure.
    # 
    # Key: C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Let's use the rank-7 Strassen decomposition for the 2x2x2 sub-block
    # where i in {0,1}, j in {0,1}, k in {0,1}.
    # Then handle the rest.
    #
    # Actually, let me just implement a verified rank-48 using a smart decomposition.
    # 
    # For 3x3x8: I'll use the fact that we can do 4x4x4 with rank 9 (Strassen variant).
    # Embed 3x3x8 into 4x4x8, use 4x4x4 rank 9 twice = 18, plus corrections.
    # This is getting complex. Let me just do a careful direct construction.
    
    # Direct rank-36 construction using algebraic identities:
    # For each pair (i,k), C[i,k] = sum_j A[i,j]B[j,k]
    # 
    # Use the identity: for 3 columns, C[i,0]+C[i,1]+C[i,2] related terms.
    # 
    # Let me try: construct using 3x3x3 rank 23 + 3x3x5 rank 15 = 38?
    # 3x3x5: schoolbook 45, but maybe rank 15? Unlikely.
    
    # Let me just implement a verified approach. Start with rank 48.
    
    # Construction: For each i in 0..2, k in 0..7:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # 
    # Group by i: for each i, we have 8 outputs, each is a dot product of
    # a 3-vector (A[i,:]) with 8 vectors (B[:,k]).
    # 
    # This is essentially 3 independent 1x3x8 multiplications.
    # 1x3x8 rank: at most 3*8=24, but actually rank 3 (since it's just
    # sum of 3 products per output, but outputs are independent).
    # Wait, 1x3x8: C[k] = sum_j A[j]*B[j,k] for k=0..7.
    # This is an 8-dimensional output, each requiring 3 multiplications.
    # But they share the A[j] factors.
    # Rank of 1x3x8 tensor: it's the sum over j of A[j] * (row j of B as 8-vector).
    # This has rank at most 3 (one term per j), since C = A[0]*B[0,:] + A[1]*B[1,:] + A[2]*B[2,:]
    # as a vector. So rank 3 for each i!
    #
    # Total rank: 3 (for i=0) + 3 (for i=1) + 3 (for i=2) = 9?
    # Wait, that can't be right. Let me reconsider.
    #
    # The tensor has indices (i,j) for U, (j,k) for V, (k,i) for W.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # 
    # For fixed i: T[i*m+j, j*p+k, k*n+i] = 1 for all j,k.
    # This means: for fixed i, the terms are (j, j*p+k) -> (k*n+i).
    # U-index: i*m+j (varies with j), V-index: j*p+k (varies with j,k), W-index: k*n+i (varies with k, fixed i).
    #
    # For fixed i, we have a sub-tensor S_i[j, j*p+k, k*n+i] = 1.
    # This is: for j in 0..2, k in 0..7: U[i*m+j], V[j*p+k], W[k*n+i] all get 1.
    #
    # Can this be rank 3? We need: S_i = sum_{r=1}^{3} u_r ⊗ v_r ⊗ w_r
    # where w_r[k*n+i] = 1 for all k (since W-index is k*n+i, and i is fixed).
    # So w_r is a vector that is 1 at positions k*n+i for k=0..7, and 0 elsewhere.
    # That's the same w for all r! So w is fixed.
    # Then we need S_i[j, v_idx] = sum_r u_r[j] * v_r[v_idx] * w[v_idx]
    # Since w[k*n+i] = 1, this is: for each k, S_i[j, j*p+k] = sum_r u_r[j] * v_r[j*p+k]
    #
    # We need: for all j,k: sum_r u_r[j] * v_r[j*p+k] = 1
    # This is: the 3x8 matrix M[j,k] = u[j] * v_r[j*p+k] summed over r.
    # Let U_vec = [u_1, u_2, u_3] (3x3 matrix), V_vec = [v_1, v_2, v_3] (each v_r is 24-dim, but only positions j*p+k matter).
    # 
    # Actually, v_r[j*p+k] for fixed r: this is a 3x8 matrix V_r.
    # We need: sum_r U_vec[:,j] * V_r[j,k] = 1 for all j,k.
    # I.e., U_vec^T * (sum over j of ... ) 
    #
    # More precisely: (U_vec^T V_r)[j,k] summed over r = 1 for all j,k.
    # Where (U_vec^T V_r)[j,k] = u_r[j] * V_r[j,k].
    #
    # Let A = [u_1, u_2, u_3] be 3x3 (rows j, cols r).
    # Let B_r be 3x8 for each r.
    # We need: sum_r A[:,r] * B_r[j,k] = 1 for all j,k.
    # I.e., sum_r (A[:,r] ⊗ B_r) = J (3x8 all-ones), where A[:,r] is a 3-vector.
    # This is: sum_r outer(A[:,r], B_r[j,k]) ... 
    #
    # For each k: sum_r A[:,r] * B_r[:,k] = 1_vector (3-vector of ones).
    # I.e., A * [B_1[:,k], B_2[:,k], B_3[:,k]] = 1_vector for each k.
    # Let C_k = [B_1[:,k], B_2[:,k], B_3[:,k]] be 3x3.
    # Then A * C_k = 1 for all k.
    # If A is invertible, C_k = A^{-1} * 1 for all k.
    # So B_r[:,k] is the r-th column of A^{-1} * 1, same for all k!
    # 
    # This works! Rank 3 per i, total rank 9? But that seems too low.
    # Let me verify: if B_r[:,k] = c_r (constant in k), then v_r[j*p+k] = c_r for all k.
    # But v_r is a single vector indexed by (j*p+k). For different j, v_r[j*p+k] can differ.
    # 
    # Wait, I made an error. v_r[j*p+k] depends on both j and k.
    # Let me redo: for fixed r, v_r is a vector of length m*p = 24.
    # v_r[j*p+k] for j in 0..2, k in 0..7.
    #
    # We need: for all j, k: sum_r u_r[j] * v_r[j*p+k] = 1.
    #
    # Let U = [u_1, u_2, u_3] (3x3, entry (j,r) = u_r[j]).
    # For each j, define vector v_r[j,:] of length 8: v_r[j*p+k] for k=0..7.
    # Then for each j: sum_r u_r[j] * v_r[j,:] = 1_vector (length 8).
    # I.e., U[j,:] @ V_j = 1, where V_j is 3x8 (entry (r,k) = v_r[j*p+k]).
    #
    # If U is invertible, V_j = U^{-1} @ 1 for all j.
    # So v_r[j*p+k] = (U^{-1} @ 1)_r for all j,k.
    # This means v_r[j*p+k] is constant in both j and k!
    # v_r[j*p+k] = c_r for all j,k.
    #
    # But then v_r is a vector that is c_r at all 24 positions.
    # And u_r[j] * c_r summed over r = sum_r u_r[j] * c_r = (U @ c)_j.
    # We need (U @ c)_j = 1 for all j, so c = U^{-1} @ 1.
    #
    # This works! So for fixed i, rank 3 suffices!
    # Total rank = 3 * 3 = 9?
    #
    # Wait, but the W-index is k*n+i. For different i, the W-indices are different.
    # For i=0: W-index = k*3+0 = 0,3,6,9,...,21
    # For i=1: W-index = k*3+1 = 1,4,7,10,...,22
    # For i=2: W-index = k*3+2 = 2,5,8,11,...,23
    #
    # These are disjoint! So the W-vectors for different i have support on disjoint sets.
    # Thus the total rank is 3+3+3 = 9.
    #
    # But wait, the U and V indices overlap across different i!
    # For i=0: U-index = 0*m+j = j (j=0,1,2), V-index = j*p+k (j=0,1,2, k=0..7)
    # For i=1: U-index = 1*m+j = 3+j (j=0,1,2), V-index = j*p+k (same as i=0!)
    # For i=2: U-index = 2*m+j = 6+j, V-index = j*p+k (same!)
    #
    # So V-indices are shared across different i! This means the v_r vectors must work for all i simultaneously.
    #
    # For i=0: need sum_r u_r^0[j] * v_r^0[j*p+k] = 1
    # For i=1: need sum_r u_r^1[j] * v_r^1[j*p+k] = 1
    # But v_r must be the same vector (since it's a single V matrix column)!
    #
    # Unless... we use different rank terms for each i. The total rank is the number of columns in U,V,W.
    # For i=0, we use 3 terms (columns 0,1,2).
    # For i=1, we use 3 terms (columns 3,4,5).
    # For i=2, we use 3 terms (columns 6,7,8).
    # Total rank 9.
    #
    # For column r (0<=r<3), corresponding to i=0:
    #   U[i*m+j, r] = u_r^0[j] for i=0, else 0.
    #   V[j*p+k, r] = v_r^0[j*p+k] for all j,k.
    #   W[k*n+i, r] = 1 if i=0 (i.e., W[k*3+0, r]=1), else 0.
    #
    # For column r (3<=r<6), corresponding to i=1:
    #   U[i*m+j, r] = u_r^1[j] for i=1, else 0.
    #   V[j*p+k, r] = v_r^1[j*p+k] for all j,k.
    #   W[k*n+i, r] = 1 if i=1, else 0.
    #
    # Etc.
    #
    # Now, the V-entries: for column r (i=0), V[j*p+k, r] = c_r (constant, as derived).
    # For column r' (i=1), V[j*p+k, r'] = c_r' (constant).
    #
    # So V[j*p+k, r] = c_r for all j,k, where c_r is a constant per column.
    #
    # And U[i*m+j, r] = u_r[i_offset] * delta_{i, i_offset} where i_offset = r//3.
    # Specifically, for r = 3*i_offset + s (s=0,1,2):
    #   U[i*m+j, r] = u_s^i[j] if i == i_offset, else 0.
    #
    # And W[k*n+i, r] = 1 if i == i_offset, else 0. (independent of k!)
    #
    # Let me verify: T[i*m+j, j*p+k, k*n+i'] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i',r]
    # = sum_r U[i*m+j,r] * c_r * W[k*n+i',r]
    #
    # For the term to be nonzero: i must equal i_offset (from U), and i' must equal i_offset (from W).
    # So i = i' = i_offset.
    # Then sum over s: U[i*m+j, 3*i+s] * c_{3*i+s} * W[k*n+i, 3*i+s]
    # = sum_s u_s^i[j] * c_s * 1
    # = (sum_s u_s^i[j] * c_s)
    # We need this = 1 for all j.
    #
    # So we need: for each i, the vector u^i[j] = sum_s u_s^i[j] * c_s = 1 for all j.
    # I.e., u^i = 1_vector.
    #
    # So we can simply set: u_s^i[j] = 1/3 for all s,j (and all i), and c_s = 1 for all s.
    # Then sum_s u_s^i[j] * c_s = 3 * (1/3) * 1 = 1. 
    #
    # Let me double-check with concrete values:
    # n=3, m=3, p=8.
    # Rank R=9.
    # 
    # For r=0 (i_offset=0, s=0):
    #   U[i*m+j, 0] = 1/3 if i==0, else 0.
    #   V[j*p+k, 0] = 1 for all j,k.
    #   W[k*n+i, 0] = 1 if i==0, else 0. (i.e., W[0,0]=1, W[3,0]=1, ..., W[21,0]=1)
    #
    # For r=1 (i_offset=0, s=1):
    #   U[i*m+j, 1] = 1/3 if i==0, else 0.
    #   V[j*p+k, 1] = 1.
    #   W[k*n+i, 1] = 1 if i==0, else 0.
    #
    # For r=2 (i_offset=0, s=2): similar.
    #
    # For r=3 (i_offset=1, s=0):
    #   U[i*m+j, 3] = 1/3 if i==1, else 0.
    #   V[j*p+k, 3] = 1.
    #   W[k*n+i, 3] = 1 if i==1, else 0.
    #
    # Etc.
    #
    # Verify T[0*3+0, 0*8+0, 0*3+0] = T[0,0,0]:
    # = sum_r U[0,r] V[0,r] W[0,r]
    # r=0: U[0,0]=1/3, V[0,0]=1, W[0,0]=1 -> 1/3
    # r=1: U[0,1]=1/3, V[0,1]=1, W[0,1]=1 -> 1/3
    # r=2: U[0,2]=1/3, V[0,2]=1, W[0,2]=1 -> 1/3
    # r=3: U[0,3]=0 -> 0
    # ...
    # Total: 1/3+1/3+1/3 = 1. Correct!
    #
    # Verify T[0*3+0, 0*8+1, 1*3+0] = T[0,1,3]:
    # This should be 0 (since k=1 in V-index means k=1, but W-index 3 means k=1,i=0... wait)
    # V-index = j*p+k = 0*8+1 = 1, so j=0,k=1.
    # W-index = k*n+i = 1*3+0 = 3, so k=1,i=0.
    # U-index = i*m+j = 0*3+0 = 0, so i=0,j=0.
    # We need i(i from U)=i(W)=0, k(V)=k(W)=1, j(U)=j(V)=0. All consistent!
    # So T[0,1,3] should be 1.
    #
    # = sum_r U[0,r] V[1,r] W[3,r]
    # r=0: U[0,0]=1/3, V[1,0]=1, W[3,0]=1 -> 1/3
    # r=1: 1/3, r=2: 1/3. Total 1. Correct!
    #
    # Verify T[0*3+0, 1*8+0, 0*3+1] = T[0,8,1]:
    # U-index 0: i=0,j=0. V-index 8: j=1,k=0. W-index 1: k=0,i=1.
    # j mismatch: j(U)=0, j(V)=1. So T should be 0.
    # = sum_r U[0,r] V[8,r] W[1,r]
    # r=0: U[0,0]=1/3, V[8,0]=1, W[1,0]=0 (W[1,0]: i=1, but r=0 has i_offset=0, so W[1,0]=0) -> 0
    # All r with i_offset=0: W[1,r]=0. r with i_offset=1: U[0,r]=0.
    # Total: 0. Correct!
    #
    # This works! Rank 9 for 3x3x8!
    #
    # But wait, this seems almost too good. Let me double-check the tensor definition.
    # T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' and i=i', k=k', j=j'.
    # In my construction, I'm getting T[i*m+j, j*p+k, k*n+i] = 1, which is correct.
    
    # Let me implement and verify this rank-9 decomposition.
    
    R = 9
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    
    for i_offset in range(n):
        for s in range(3):
            r = i_offset * 3 + s
            # U: 1/3 for rows i_offset*m + j, all j
            for j in range(m):
                U[i_offset * m + j, r] = 1.0 / 3.0
            # V: 1 for all entries
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = 1.0
            # W: 1 for rows k*n + i_offset, all k
            for k in range(p):
                W[k * n + i_offset, r] = 1.0
    
    # Verify
    if _verify(U, V, W):
        if time.time() < deadline:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 72
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_sb), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
