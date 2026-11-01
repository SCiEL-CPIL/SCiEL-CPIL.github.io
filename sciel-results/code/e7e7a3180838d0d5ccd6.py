"""
Refines the constructive block decomposition strategy for 2x4 * 4x7 matrix multiplication.
Uses Lifting/Subtensor analysis: the 1x4x7 sub-tensor (fixing i=0) has rank 28 (schoolbook).
The full 2x4x7 tensor requires additional terms for i=1 that are linearly independent from the i=0 terms.

Strategy: Use Winograd-style Fourier transform technique on the p=7 dimension.
Split into overlapping windows to exploit structure in the 4x7 multiplication.

Key insight: For 1x4x7, we can use a rank-28 decomposition. For the full 2x4x7,
we need to add terms for the second row of A. By careful construction, we can share
some structure between the two rows.

Current approach: Rank 50 via block decomposition with Strassen and schoolbook.
Attempting to reduce by using more efficient 1x4x7 base and lifting.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Lifting/Subtensor approach:
    # First, find a minimal-rank decomposition for the 1x4x7 sub-tensor (i=0 fixed).
    # The 1x4x7 tensor T_0[j,k] = A[0,j] * B[j,k] has rank 28 (schoolbook is optimal for 1xm*p when min(m,p) >= 2).
    # Actually, for 1xm*p, the tensor rank is exactly m*p since each output C[k] depends on all m inputs independently.
    # So rank(1x4x7) = 4*7 = 28.
    
    # For the full 2x4x7 tensor, we need rank >= 28 (from the i=0 subspace).
    # The additional terms for i=1 must be linearly independent from the i=0 terms in the W-space.
    
    # Strategy: Use a structured decomposition that shares V-factors between i=0 and i=1 where possible.
    # Key: C[i,k] = sum_j A[i,j] * B[j,k]. For fixed k, C[0,k] and C[1,k] are both linear in B[:,k].
    # We can write: [C[0,k]; C[1,k]] = A[:,j] outer B[j,k] summed over j.
    
    # Alternative: Group by k. For each k, we need to compute a 2-vector: [sum_j A[0,j]*B[j,k], sum_j A[1,j]*B[j,k]].
    # This is equivalent to multiplying a 2x4 matrix A by a 4x1 vector B[:,k].
    # The rank of this 2x4x1 tensor is 8 (schoolbook) or can it be less?
    # For 2xm x 1: rank is 2m if we need exact computation, unless we use Strassen-like tricks.
    # Actually 2x4x1: C[i] = sum_j A[i,j] B[j]. This is just 8 scalar products. Rank = 8.
    
    # But we can do better by grouping k values. Consider 2x4 x 2x1 (two k values):
    # C[i,k] for k in {k0, k0+1}. This is a 2x2 x 4x2 tensor... no, it's 2x4x2.
    # Rank of 2x4x2: schoolbook is 16. Can we do better?
    # Use Strassen on the 2x2 block: split A into 2x2 blocks, B into 2x2 blocks.
    # A is 2x4 = two 2x2 blocks. B is 4x2 = two 2x2 blocks stacked.
    # C = A*B = [A_left * B_top + A_right * B_bottom] (2x2 result for each of 2 k-values... wait)
    
    # Let me reconsider. For 2x4 * 4x2 -> 2x2:
    # A = [A1 | A2] where A1, A2 are 2x2. B = [B1; B2] where B1, B2 are 2x2.
    # C = A1*B1 + A2*B2. Each 2x2*2x2 uses Strassen (rank 7). Total rank = 14.
    # Schoolbook would be 2*4*2 = 16. So Strassen saves 2.
    
    # For 2x4 * 4x1 -> 2x1: schoolbook is 8. No Strassen benefit (no 2x2 block in B).
    
    # So for p=7: split into three 2x2 blocks (k=0,1; 2,3; 4,5) and one 1-block (k=6).
    # Three 2x4x2 subproblems: each rank 14 (Strassen on 2x2*2x2 twice). Total: 42.
    # One 2x4x1 subproblem: rank 8 (schoolbook). Total: 50.
    
    # This matches the current rank 50. Can we do better?
    
    # Idea: Use overlapping windows or Winograd technique.
    # For 1x4x7, the minimal rank is 28. For 2x4x7, lower bound is 28.
    # Upper bound from current: 50.
    
    # Try: Can we share structure between adjacent k-blocks?
    # Winograd Fourier Transform Technique for matrix multiplication.
    # For 1x4x7: use cyclic convolution structure.
    
    # Actually, let's try a different split: use 3x3 blocks with Strassen-like optimization.
    # Or use the fact that 7 = 4 + 3, and handle 4x4 and 4x3 separately.
    
    # For 2x4 * 4x4 -> 2x4: 
    # Split A into two 2x2, B into four 2x2. C = sum of 2x2*2x2 products.
    # Each 2x2*2x2 is Strassen rank 7. Four of them: rank 28.
    # Schoolbook: 2*4*4 = 32. Save 4.
    
    # For 2x4 * 4x3 -> 2x3:
    # Split into 2x4*4x2 (rank 14 as above) + 2x4*4x1 (rank 8) = 22.
    # Schoolbook: 2*4*3 = 24. Save 2.
    
    # Total: 28 + 22 = 50. Same as before.
    
    # Try 2x4*4x7 directly with a different strategy:
    # Use rank-1 updates. Each term is u_i * v_j * w_k.
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j,k} (the multiplication tensor).
    
    # Lower bound: For 2x4x7, the slice rank lower bound... 
    # The tensor has 2*4*7 = 56 non-zero entries. Each rank-1 term can cover at most... 
    # Actually, the rank is at least the max over all slices of the rank of that slice.
    # Slice fixing i=0: 4x7 matrix, rank 4 (full). So rank >= 4.
    # Slice fixing j=0: 2x7 matrix, rank 2. So rank >= 2.
    # Slice fixing k=0: 2x4 matrix, rank 2. So rank >= 2.
    # These are weak.
    
    # Better lower bound: border rank or geometric rank.
    # For matmul n x m x p, known lower bounds: max(n,m,p) * something.
    
    # Let's just try to verify if rank 48 or 49 is possible with a different construction.
    
    # Alternative: Use 3x3 Strassen-like for some blocks.
    # 3x3*3x3 has rank 23 (known optimal). But our blocks are 2x2.
    
    # Try: Instead of splitting p=7 into 2+2+2+1, use 3+2+2.
    # 2x4*4x3: as above, rank 22.
    # 2x4*4x2: rank 14.
    # 2x4*4x2: rank 14.
    # Total: 22+14+14 = 50. Same.
    
    # Try 4+3: 
    # 2x4*4x4: rank 28.
    # 2x4*4x3: rank 22.
    # Total: 50. Same.
    
    # The structure seems to force rank 50 with this block approach.
    
    # Can we use a global Strassen-like decomposition?
    # 2x4*4x7: view as 2x4*4x7 = (2x2 + 2x2) * (4x7).
    # Not directly helpful.
    
    # Try: Pad to 2x4*4x8 and use 4x4 blocks? 8=4+4.
    # 2x4*4x4: rank 28. 2x4*4x4: rank 28. Total: 56. Worse.
    
    # The current rank 50 appears tight for this family.
    # Keep the existing construction which achieves rank 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a Strassen decomposition for a 2x2 (A block) * 2x2 (B block) -> 2x2 (C block).
        A indices: i0, i0+1 rows; j0, j0+1 cols.
        B indices: j0, j0+1 rows; k0, k0+1 cols.
        C indices: i0, i0+1 rows; k0, k0+1 cols.
        """
        nonlocal col
        
        # A indices (row-major in U: i*m + j)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices (row-major in V: j*p + k)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices (row-major in W: k*n + i)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # Strassen products
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        """
        Adds a Schoolbook decomposition for a 2x2 (A block) * 2x1 (B block) -> 2x1 (C block).
        """
        nonlocal col
        
        # A indices
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # W indices
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        
        # P2 = b*g
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        
        # P3 = c*e
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # P4 = d*g
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    # Assemble the decomposition
    # Two j-blocks: j_start in [0, 2]
    for j_start in [0, 2]:
        # Three k-blocks of size 2: k_start in [0, 2, 4]
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        # One k-block of size 1: k_start = 6
        add_schoolbook_221_block(0, j_start, 6)

    # Verify correctness
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Fast verification using numpy
    def verify(U, V, W):
        r = U.shape[1]
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We check T[i*m+j, j*p+k, k*n+i] == 1 for all valid (i,j,k), and 0 otherwise.
        Texp = np.zeros((n*m, m*p, p*n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Texp[i*m + j, j*p + k, k*n + i] = 1
        
        T = np.zeros((n*m, m*p, p*n), dtype=int)
        for c in range(r):
            T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        return np.array_equal(T, Texp)

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
