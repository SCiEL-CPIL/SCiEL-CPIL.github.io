"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Verify helper
    def verify(U, V, W, n, m, p, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition (rank = 96)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Hybrid decomposition:
    # 1. Rows i=0,1 (2x4x8): Use a decomposition of rank 14.
    #    We can view this as two 1x4x8 decompositions coupled, or use a known 2x4x8 algorithm.
    #    Actually, a simpler robust approach for 2x4x8:
    #    We can use the "middle product" structure or just combine two 1x4x8 if we can't find a better 2x4x8.
    #    Rank of 1x4x8 is 8. So two rows would be 16.
    #    Can we do better? 
    #    Let's try to construct a specific rank-14 decomposition for 2x4x8 if possible, 
    #    or just use rank 16 for the first two rows and rank 8 for the third, total 24.
    #    Wait, the prompt suggests rank 22. 
    #    Let's stick to a verifiable construction.
    #    
    #    Construction for 2x4x8:
    #    We can decompose the 4x8 matrix multiplication for 2 rows.
    #    Actually, let's just use the sum of two rank-8 decompositions for the first two rows?
    #    Row 0: rank 8. Row 1: rank 8. Total 16.
    #    Row 2: rank 8. Total 24.
    #    
    #    Is there a rank 22?
    #    The lower bound for 3x4x8 is not trivially 22.
    #    However, we can try to optimize the 2x4x8 part.
    #    If we treat the 2x4 matrix as a single vector of 8 elements and 4x8 as 32 elements...
    #    Actually, let's just implement the rank-24 solution (2 * rank(1x4x8) + rank(1x4x8)) = 24.
    #    This is a massive improvement over 96.
    #    
    #    Let's check if we can do better than 24.
    #    For 1x4x8, the rank is 8 (since it's just a dot product for each of the 8 outputs, 
    #    and the 4 inputs are shared? No.
    #    T[i=0, j, k] = A[0, j] * B[j, k].
    #    This is a rank-8 tensor?
    #    U[0, j] = A[0, j]. V[j, k] = B[j, k]. W[k, 0] = 1.
    #    For each k, we need to sum over j.
    #    The tensor is T_{j,k} = A_j B_{jk}.
    #    This can be written as sum_{c=1}^8 U_c V_c W_c.
    #    Let's use the standard basis for the 4x8 product.
    #    Actually, the matrix multiplication tensor for 1x4x8 has rank 8?
    #    No, the rank of the matrix multiplication tensor MM_{n,m,p} is at least n*m*p / something.
    #    For 1xm p, the rank is m*p? No.
    #    Let's just use the schoolbook for 1x4x8 which is 4*8 = 32 terms?
    #    No, for 1x4x8:
    #    C_k = sum_j A_j B_jk.
    #    This is a bilinear form.
    #    We can decompose this into 8 rank-1 terms?
    #    Let's try:
    #    Term r (for r in 0..7 corresponding to k=r):
    #    U[j] = A_j (for all j? No, U is a vector of length 4).
    #    V[j, k] = B_jk.
    #    W[k] = 1 if k==r else 0.
    #    This doesn't work directly because U and V must be fixed vectors for the tensor decomposition.
    #    
    #    Let's use the known fact: Rank(MM_{1,m,p}) = m*p? 
    #    Actually, for 1x1x1 rank is 1.
    #    For 1x2x2: C0 = A0B00 + A1B10, C1 = A0B01 + A1B11.
    #    Can we do it in rank 2?
    #    T = sum_{j=0}^1 A_j (B_j0 e_0 + B_j1 e_1).
    #    This is sum_{j=0}^1 A_j B_j0 e_0 + sum_{j=0}^1 A_j B_j1 e_1.
    #    This is 4 terms. Rank 4?
    #    Wait, the tensor is T_{j,k} = A_j B_{jk}.
    #    This is a matrix of size 4x8 (flattened).
    #    The tensor is rank 1 if B is rank 1. But B is a variable.
    #    The tensor is the "multiplication tensor".
    #    For 1xm p, the tensor is T = sum_{j=1}^m A_j \otimes B_{j, .} \otimes I_p? No.
    #    
    #    Let's just use the schoolbook decomposition for each row separately.
    #    For row i, we need to compute C_i = A_i B.
    #    This is a 1xm p multiplication.
    #    The rank of MM_{1,m,p} is m*p.
    #    So for 3 rows, total rank is 3 * 4 * 8 = 96.
    #    This is the same as schoolbook.
    #    
    #    So my previous "rank 22" idea was flawed because I confused the rank of the matrix multiplication *algorithm* with the tensor rank.
    #    Strassen reduces the number of *multiplications* in an algorithm, but the tensor rank of the *entire* multiplication tensor is what we are minimizing.
    #    However, Strassen *does* provide a tensor decomposition of lower rank than schoolbook.
    #    For 2x2x2, rank is 7 (Strassen) vs 8 (Schoolbook).
    #    
    #    So we should use Strassen for the 2x2x2 blocks.
    #    The current program attempts this but has bugs.
    #    
    #    Let's fix the Strassen block implementation.
    #    Dimensions: n=3, m=4, p=8.
    #    We can partition into 2x2x2 blocks.
    #    i: 0,1 (2 rows), 2 (1 row)
    #    j: 0,1 (2 cols), 2,3 (2 cols)
    #    k: 0,1 (2 cols), 2,3 (2 cols), 4,5 (2 cols), 6,7 (2 cols)
    #    
    #    Blocks:
    #    (0,2) x (0,2) x (0,2) -> 2x2x2 -> Rank 7
    #    (0,2) x (0,2) x (2,4) -> 2x2x2 -> Rank 7
    #    (0,2) x (0,2) x (4,6) -> 2x2x2 -> Rank 7
    #    (0,2) x (0,2) x (6,8) -> 2x2x2 -> Rank 7
    #    (0,2) x (2,4) x (0,2) -> 2x2x2 -> Rank 7
    #    (0,2) x (2,4) x (2,4) -> 2x2x2 -> Rank 7
    #    (0,2) x (2,4) x (4,6) -> 2x2x2 -> Rank 7
    #    (0,2) x (2,4) x (6,8) -> 2x2x2 -> Rank 7
    #    (2,3) x (0,2) x (0,2) -> 1x2x2 -> Rank 4 (Schoolbook)
    #    (2,3) x (0,2) x (2,4) -> 1x2x2 -> Rank 4
    #    (2,3) x (0,2) x (4,6) -> 1x2x2 -> Rank 4
    #    (2,3) x (0,2) x (6,8) -> 1x2x2 -> Rank 4
    #    (2,3) x (2,4) x (0,2) -> 1x2x2 -> Rank 4
    #    (2,3) x (2,4) x (2,4) -> 1x2x2 -> Rank 4
    #    (2,3) x (2,4) x (4,6) -> 1x2x2 -> Rank 4
    #    (2,3) x (2,4) x (6,8) -> 1x2x2 -> Rank 4
    #    
    #    Total Rank = 8 * 7 + 8 * 4 = 56 + 32 = 88.
    #    This is the same as the current program's target, but the current program has bugs in the Strassen implementation.
    #    
    #    Let's implement a correct Strassen block.
    
    def build_strassen_block(i0, j0, k0, U, V, W, col):
        """Add 7 Strassen terms for the 2x2x2 block starting at (i0,j0,k0)."""
        # Indices within block:
        # i, j, k in {0, 1}
        # Global indices: I = i0+i, J = j0+j, K = k0+k
        # U index: I*m + J
        # V index: J*p + K
        # W index: K*n + I
        
        # Strassen terms for 2x2x2:
        # p1 = (A00+A11)(B00+B11)
        # p2 = (A10+A11)B00
        # p3 = A00(B10-B11)
        # p4 = A11(B01-B10)
        # p5 = (A00+A01)B11
        # p6 = (A10-A00)(B00+B10)
        # p7 = (A01-A11)(B01+B11)
        
        # C00 = p1 - p5 + p6 - p7
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p3 + p4 + p7
        
        # Term 1: p1
        U[(i0+0)*m+(j0+0)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+0)*p+(k0+0)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[(k0+0)*n+(i0+0)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # Term 2: p2
        U[(i0+1)*m+(j0+0)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+0)*p+(k0+0)][col] += 1
        W[(k0+0)*n+(i0+1)][col] += 1
        # C10 gets +1, C11 gets -1?
        # C10 = p2 + p4. So W[(k0+0)*n+(i0+1)] += 1.
        # C11 = p1 - p3 + p4 + p7. p2 is not in C11.
        # Wait, standard Strassen:
        # C11 = p1 - p3 + p4 + p7.
        # So p2 only contributes to C10.
        W[(k0+0)*n+(i0+1)][col] += 1
        col += 1
        
        # Term 3: p3
        U[(i0+0)*m+(j0+0)][col] += 1
        V[(j0+0)*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += -1
        # C01 = p3 + p5. So W[(k0+1)*n+(i0+0)] += 1.
        # C11 = p1 - p3 + p4 + p7. So W[(k0+1)*n+(i0+1)] += -1.
        W[(k0+1)*n+(i0+0)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += -1
        col += 1
        
        # Term 4: p4
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+0)*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+0)][col] += -1
        # C10 = p2 + p4. So W[(k0+0)*n+(i0+1)] += 1.
        # C11 = p1 - p3 + p4 + p7. So W[(k0+1)*n+(i0+1)] += 1.
        W[(k0+0)*n+(i0+1)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # Term 5: p5
        U[(i0+0)*m+(j0+0)][col] += 1
        U[(i0+0)*m+(j0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        # C00 = p1 - p5 + p6 - p7. So W[(k0+0)*n+(i0+0)] += -1.
        # C01 = p3 + p5. So W[(k0+1)*n+(i0+0)] += 1.
        W[(k0+0)*n+(i0+0)][col] += -1
        W[(k0+1)*n+(i0+0)][col] += 1
        col += 1
        
        # Term 6: p6
        U[(i0+1)*m+(j0+0)][col] += 1
        U[(i0+0)*m+(j0+0)][col] += -1
        V[(j0+0)*p+(k0+0)][col] += 1
        V[(j0+1)*p+(k0+0)][col] += 1
        # C00 = p1 - p5 + p6 - p7. So W[(k0+0)*n+(i0+0)] += 1.
        W[(k0+0)*n+(i0+0)][col] += 1
        col += 1
        
        # Term 7: p7
        U[(i0+0)*m+(j0+1)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += -1
        V[(j0+0)*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        # C00 = p1 - p5 + p6 - p7. So W[(k0+0)*n+(i0+0)] += -1.
        # C11 = p1 - p3 + p4 + p7. So W[(k0+1)*n+(i0+1)] += 1.
        W[(k0+0)*n+(i0+0)][col] += -1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        return col

    def build_block_decomposition():
        i_blocks = [(0, 2), (2, 3)]
        m_blocks = [(0, 2), (2, 4)]
        p_blocks = [(0, 2), (2, 4), (4, 6), (6, 8)]
        
        total_rank = 0
        blocks_info = []
        for i0, i1 in i_blocks:
            for j0, j1 in m_blocks:
                for k0, k1 in p_blocks:
                    di = i1 - i0
                    dj = j1 - j0
                    dk = k1 - k0
                    if di == 2 and dj == 2 and dk == 2:
                        r = 7
                        method = 'strassen'
                    else:
                        r = di * dj * dk
                        method = 'schoolbook'
                    total_rank += r
                    blocks_info.append((i0, i1, j0, j1, k0, k1, method, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, i1, j0, j1, k0, k1, method, r) in blocks_info:
            di = i1 - i0
            dj = j1 - j0
            dk = k1 - k0
            
            if method == 'strassen':
                col = build_strassen_block(i0, j0, k0, U, V, W, col)
            else:
                for di_idx in range(di):
                    for dj_idx in range(dj):
                        for dk_idx in range(dk):
                            U[(i0+di_idx)*m + (j0+dj_idx)][col] = 1
                            V[(j0+dj_idx)*p + (k0+dk_idx)][col] = 1
                            W[(k0+dk_idx)*n + (i0+di_idx)][col] = 1
                            col += 1
        
        if col != total_rank:
            return None, None, None, 0
        
        return U, V, W, total_rank

    # Try block decomposition
    U, V, W, rank = build_block_decomposition()
    
    if U is not None and verify(U, V, W, n, m, p, rank):
        print(f"Block decomposition verified with rank {rank}")
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    if verify(U, V, W, n, m, p, rank):
        print(f"Schoolbook decomposition verified with rank {rank}")
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
