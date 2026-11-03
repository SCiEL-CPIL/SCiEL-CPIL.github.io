import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Rank-14 solution: Split m=4 into two halves of size 2.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each 2x2 @ 2x8 product has rank 4 (naive: one rank-1 term per (i,k) pair).
    # Total rank = 2 * 4 = 8? No, let's think carefully.
    #
    # For a single 2x2 @ 2x8: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # Tensor T[i,j,k] = 1 for all i in {0,1}, j in {0..7}, k in {0,1}
    # This tensor = sum over i,k of e_i (in A-space) * 1 (in B-space) * e_k... 
    # Actually T[i,j,k] = 1 means the tensor is 1_2 (A) x 1_8 (B) x 1_2 (K)?
    # No. The multiplication tensor for A(2x2) * B(2x8) = C(2x8):
    # T is in R^{2 x 8 x 2} (A-index x C-index x B-index, or whatever convention)
    # T[a,c,b] = 1 if C[c] uses A[a] and B[b], i.e., T[a,c,b] = 1 for all a,c,b
    # So T = 1_2 (A) tensor 1_8 (C) tensor 1_2 (B) which has rank 1!
    # But that can't be right because rank-1 would mean C = (sum A)(sum B) which is wrong.
    #
    # I need to be more careful about the tensor convention.
    # The multiplication tensor M satisfies: C = A*B iff
    # C[c] = sum_{a,b} M[a,c,b] * A[a] * B[b]
    # For 2x2 * 2x8: C has 2*8=16 entries, A has 4 entries, B has 16 entries.
    # C[i,j] = sum_{k=0}^{1} A[i,k] * B[k,j]
    # So M[i,j,k] = 1 for all i in {0,1}, j in {0..7}, k in {0,1}
    # M is a 2 x 8 x 2 tensor with all entries 1.
    # M = (sum_i e_i) tensor (sum_j e_j) tensor (sum_k e_k) = 1_2 x 1_8 x 1_2
    # This has tensor rank 1!
    #
    # But wait, that means C[i,j] = (sum_k A[i,k]) * (sum_k' B[k',j])... NO!
    # A rank-1 tensor M = u tensor v tensor w gives C[i,j] = u[i]*v[j]*w[k] summed...
    # I'm confusing the contraction. Let me think step by step.
    #
    # Tensor decomposition: M = sum_r u_r tensor v_r tensor w_r
    # Then C[i,j] = sum_{k} M[i,j,k] * A[i,k] * B[k,j]... no that's not right either.
    #
    # The correct formulation: C[i,j] = sum_{a,b} M[a,i,j,b] * A[a] * B[b]
    # where a indexes A entries, b indexes B entries, (i,j) indexes C entries.
    # For 2x2 * 2x8: a in {0,1,2,3} (row,col of A), b in {0..15} (row,col of B),
    # (i,j) in {0..15} (row,col of C).
    # M[a, (i,j), b] = 1 if A[a] and B[b] contribute to C[i,j].
    # A[a] = A[row(a), col(a)] where a = row*2+col
    # B[b] = B[row(b), col(b)] where b = row*8+col
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So M[a, (i,j), b] = 1 iff row(a)=i, col(b)=j, and col(a)=row(b)
    # i.e., a = i*2+k, b = k*8+j for some k in {0,1}
    # M is 4 x 16 x 16.
    #
    # For a fixed (i,j): M[a,(i,j),b] = 1 when a=i*2+k and b=k*8+j for k=0,1
    # So the slice M[:,(i,j),:] has ones at positions (i*2, j) and (i*2+1, 8+j)
    # This slice is the sum of two rank-1 vectors: e_{i*2} tensor e_j + e_{i*2+1} tensor e_{8+j}
    #
    # The full tensor M = sum_{i=0}^{1} sum_{j=0}^{7} sum_{k=0}^{1} 
    #   e_{i*2+k} (in A-space) tensor e_{i*8+j} (in C-space) tensor e_{k*8+j} (in B-space)
    # This is a sum of 2*8*2 = 32 rank-1 terms, but many may combine.
    #
    # Actually, let's just use the naive decomposition with rank = n*m = 8:
    # For each (i,k) pair in A, create one rank-1 term:
    # u_r has 1 at position i*m+k (A index), v_r has 1 at position k*p+j for all j (B row k),
    # w_r has 1 at position j*n+i for all j (C column i... wait, C index)
    #
    # Let me just implement the naive rank-8 and verify numerically.
    
    total_rank = 8  # n * m = 2 * 4
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    r = 0
    for i in range(n):
        for k in range(m):
            a_idx = i * m + k      # index in A (flattened row-major)
            U[a_idx, r] = 1
            for j in range(p):
                b_idx = k * p + j  # index in B (flattened row-major: B[k,j] at k*p+j)
                V[b_idx, r] = 1
                c_idx = j * n + i  # W[k*n+i, r] convention: C[i,j] at flat index j*n+i
                W[c_idx, r] = 1
            r += 1
    
    # Verify: C[i,j] should equal sum_{k} A[i,k] * B[k,j]
    # Using decomposition: C[i,j] = sum_r U[:,r] . A_flat * V[:,r] . B_flat * W[:,r] . 1
    # Actually the reconstruction is: C = sum_r (U[:,r] reshaped to n x m) * (V[:,r] reshaped to m x p) * W[:,r]
    # No, that's not how it works. The tensor decomposition means:
    # C[i,j] = sum_{a,b} M[a,(i,j),b] A[a] B[b] = sum_r (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b]) * W[(i,j),r]
    # So C = sum_r (U^T A_flat) * (V^T B_flat) * W^T, where each term is an outer product scaled.
    # More precisely: C[i,j] = sum_r W[i*n+j, r] * (sum_a U[a,r]*A[a]) * (sum_b V[b,r]*B[b])
    
    # Let's verify numerically
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()  # row-major: A[i,k] at index i*m+k
    B_flat = B_test.flatten()  # row-major: B[k,j] at index k*p+j
    
    C_recon = np.zeros((n, p))
    for r in range(total_rank):
        u_val = U[:, r] @ A_flat
        v_val = V[:, r] @ B_flat
        for i in range(n):
            for j in range(p):
                c_idx = j * n + i
                C_recon[i, j] += W[c_idx, r] * u_val * v_val
    
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        # Fallback to rank-56 Strassen solution
        return _construct_strassen_56(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _construct_strassen_56(n, m, p):
    """Fallback: rank-56 Strassen decomposition."""
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1
    
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j_group in range(2):
        a_map = [0*4 + j_group*2, 0*4 + j_group*2+1, 1*4 + j_group*2, 1*4 + j_group*2+1]
        for k_group in range(4):
            b_map = [(j_group*2)*8 + k_group*2, (j_group*2)*8 + k_group*2+1,
                     (j_group*2+1)*8 + k_group*2, (j_group*2+1)*8 + k_group*2+1]
            c_map = [(k_group*2)*2 + 0, (k_group*2+1)*2 + 0,
                     (k_group*2)*2 + 1, (k_group*2+1)*2 + 1]
            base_r = (j_group * 4 + k_group) * 7
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    if U_local[la, l] != 0:
                        U[a_map[la], r] += U_local[la, l]
                for lb in range(4):
                    if V_local[lb, l] != 0:
                        V[b_map[lb], r] += V_local[lb, l]
                for lc in range(4):
                    if W_local[lc, l] != 0:
                        W[c_map[lc], r] += W_local[lc, l]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
