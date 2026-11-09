"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Should equal 1 if i,j,k form a valid multiplication term, else 0
    # Actually, the tensor for matrix mult is T[i,j,k] = 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Reconstruct: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # U indices: i*m+j where i=0,1 and j=0,1
    # V indices: j*p+k where j=0,1 and k=0,1
    # W indices: k*n+i where k=0,1 and i=0,1
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1.0  # a11
    U[3][0] = 1.0  # a22
    V[0][0] = 1.0  # b11
    V[3][0] = 1.0  # b22
    W[0][0] = 1.0  # c11 (k=0,i=0)
    W[3][0] = 1.0  # c22 (k=1,i=1)
    
    # M2 = (a21+a22)b11
    U[2][1] = 1.0  # a21
    U[3][1] = 1.0  # a22
    V[0][1] = 1.0  # b11
    W[2][1] = 1.0  # c21 (k=0,i=1)
    
    # M3 = a11(b12-b22)
    U[0][2] = 1.0  # a11
    V[1][2] = 1.0  # b12
    V[3][2] = -1.0 # b22
    W[1][2] = 1.0  # c12 (k=1,i=0)
    
    # M4 = a22(b21-b11)
    U[3][3] = 1.0  # a22
    V[2][3] = 1.0  # b21
    V[0][3] = -1.0 # b11
    W[0][3] = 1.0  # c11 (k=0,i=0)
    W[2][3] = 1.0  # c21 (k=0,i=1)
    
    # M5 = (a11+a12)b22
    U[0][4] = 1.0  # a11
    U[1][4] = 1.0  # a12
    V[3][4] = 1.0  # b22
    W[0][4] = -1.0 # c11 (k=0,i=0)
    W[1][4] = 1.0  # c12 (k=1,i=0)
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1.0  # a21
    U[0][5] = -1.0 # a11
    V[0][5] = 1.0  # b11
    V[1][5] = 1.0  # b12
    W[3][5] = 1.0  # c22 (k=1,i=1)
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1.0  # a12
    U[3][6] = -1.0 # a22
    V[2][6] = 1.0  # b21
    V[3][6] = 1.0  # b22
    W[0][6] = 1.0  # c11 (k=0,i=0)
    W[3][6] = 1.0  # c22 (k=1,i=1)
    
    return U, V, W

def _combine_decompositions(decs):
    """Combine multiple decompositions by concatenation."""
    total_rank = sum(len(d[0][0]) for d in decs)
    n = m = p = 2
    U = [[0.0] * total_rank for _ in range(n*m)]
    V = [[0.0] * total_rank for _ in range(m*p)]
    W = [[0.0] * total_rank for _ in range(p*n)]
    
    offset = 0
    for U_d, V_d, W_d in decs:
        r = len(U_d[0])
        for i in range(n*m):
            for rr in range(r):
                U[i][offset+rr] += U_d[i][rr]
        for i in range(m*p):
            for rr in range(r):
                V[i][offset+rr] += V_d[i][rr]
        for i in range(p*n):
            for rr in range(r):
                W[i][offset+rr] += W_d[i][rr]
        offset += r
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using Strassen for 2x2x2 blocks and schoolbook for rest."""
    n, m, p = 2, 4, 7
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # We will build the full matrices
    # Rank estimate: 4 blocks of 2x2x2 (Strassen, rank 7) + remaining schoolbook
    # A (2x4) split into A1 (2x2), A2 (2x2)
    # B (4x7) split into B1 (2x7), B2 (2x7)
    # C = A1 B1 + A2 B2
    
    # B1 (2x7) split into B1a (2x2), B1b (2x2), B1c (2x3)
    # B2 (2x7) split into B2a (2x2), B2b (2x2), B2c (2x3)
    
    # Blocks:
    # 1. A1 B1a: 2x2 * 2x2 -> Strassen (7)
    # 2. A1 B1b: 2x2 * 2x2 -> Strassen (7)
    # 3. A1 B1c: 2x2 * 2x3 -> Schoolbook (2*2*3 = 12)
    # 4. A2 B2a: 2x2 * 2x2 -> Strassen (7)
    # 5. A2 B2b: 2x2 * 2x2 -> Strassen (7)
    # 6. A2 B2c: 2x2 * 2x3 -> Schoolbook (12)
    
    total_rank = 7 + 7 + 12 + 7 + 7 + 12
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    def place_strassen(U, V, W, u_offset, v_offsets, w_offset, rank_off):
        """
        Place a 2x2x2 Strassen block into the larger matrices.
        u_offset: index in U for A[0,0] (top-left of the 2x2 A block)
        v_offsets: list of 4 indices in V for B[0,0], B[0,1], B[1,0], B[1,1]
        w_offset: index in W for C[0,0] (top-left of the 2x2 C block)
        rank_off: current rank offset
        """
        # U indices for 2x2 block: [u_offset, u_offset+1, u_offset+4, u_offset+5]
        # corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
        u_indices = [u_offset, u_offset + 1, u_offset + 4, u_offset + 5]
        
        # V indices are provided explicitly because the stride in V is p (7), not 2
        # v_offsets corresponds to B[0,0], B[0,1], B[1,0], B[1,1]
        
        # W indices for 2x2 block: [w_offset, w_offset+1, w_offset+2, w_offset+3]
        # corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
        # Note: W index is k*n + i.
        # C[0,0] -> k=0, i=0 -> 0
        # C[0,1] -> k=0, i=1 -> 1
        # C[1,0] -> k=1, i=0 -> 2
        # C[1,1] -> k=1, i=1 -> 3
        w_indices = [w_offset, w_offset + 1, w_offset + 2, w_offset + 3]
        
        for r in range(7):
            # Map local Strassen indices (0,1,2,3) to global U indices
            for local_i, global_i in enumerate(u_indices):
                U[global_i][rank_off + r] += U_s[local_i][r]
            
            # Map local Strassen indices (0,1,2,3) to global V indices
            for local_i, global_i in enumerate(v_offsets):
                V[global_i][rank_off + r] += V_s[local_i][r]
            
            # Map local Strassen indices (0,1,2,3) to global W indices
            for local_i, global_i in enumerate(w_indices):
                W[global_i][rank_off + r] += W_s[local_i][r]
        
        return rank_off + 7

    def place_schoolbook_2x2x3(U, V, W, u_offset, v_base_row, v_base_col, w_offset, rank_off):
        """
        Place a 2x2 * 2x3 schoolbook block.
        A is 2x2 at u_offset (A[0,0], A[0,1], A[1,0], A[1,1])
        B is 2x3 starting at row v_base_row, col v_base_col
        C is 2x3 starting at row 0 (relative to output block), col w_offset
        """
        # A indices:
        # A[0,0] -> u_offset
        # A[0,1] -> u_offset + 1
        # A[1,0] -> u_offset + 4
        # A[1,1] -> u_offset + 5
        u_indices = [
            u_offset,
            u_offset + 1,
            u_offset + 4,
            u_offset + 5
        ]
        
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # U index for A[i,j]
                    u_idx = u_indices[i*2 + j]
                    
                    # V index for B[j, v_base_col+k]
                    # V index = row * p + col
                    # The row in B is (v_base_row + j)
                    # The col in B is (v_base_col + k)
                    v_idx = (v_base_row + j) * p + (v_base_col + k)
                    
                    # W index for C[i, w_offset+k]
                    # W index = col * n + row
                    # The col in C is (w_offset + k)
                    # The row in C is i
                    w_idx = (w_offset + k) * n + i
                    
                    U[u_idx][rank_off] = 1.0
                    V[v_idx][rank_off] = 1.0
                    W[w_idx][rank_off] = 1.0
                    rank_off += 1
        return rank_off

    rank_off = 0
    
    # 1. A1 (rows 0-1, cols 0-1) * B1a (rows 0-1, cols 0-1)
    # A1 is at U offset 0.
    # B1a is at V rows 0-1, cols 0-1. Indices: 0, 1, 7, 8.
    # C1a is at W cols 0-1. Offset 0.
    rank_off = place_strassen(U, V, W, 0, [0, 1, 7, 8], 0, rank_off)
    
    # 2. A1 (rows 0-1, cols 0-1) * B1b (rows 0-1, cols 2-3)
    # A1 is at U offset 0.
    # B1b is at V rows 0-1, cols 2-3. Indices: 2, 3, 9, 10.
    # C1b is at W cols 2-3. Offset 2.
    rank_off = place_strassen(U, V, W, 0, [2, 3, 9, 10], 2, rank_off)
    
    # 3. A1 (rows 0-1, cols 0-1) * B1c (rows 0-1, cols 4-6)
    # A1 is at U offset 0.
    # B1c is at V rows 0-1, cols 4-6. Base row 0, Base col 4.
    # C1c is at W cols 4-6. Offset 4.
    rank_off = place_schoolbook_2x2x3(U, V, W, 0, 0, 4, 4, rank_off)
    
    # 4. A2 (rows 0-1, cols 2-3) * B2a (rows 2-3, cols 0-1)
    # A2 is at U offset 4 (since A[1,0] is index 4, A[1,1] is 5? No.)
    # A is 2x4.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A2 is cols 2-3.
    # A2[0,0] = A[0,2] = index 2
    # A2[0,1] = A[0,3] = index 3
    # A2[1,0] = A[1,2] = index 6
    # A2[1,1] = A[1,3] = index 7
    # So u_offset for A2 is 2?
    # My place_strassen assumes u_offset is A[0,0] of the block.
    # For A2, A[0,0] of the block is A[0,2] which is index 2.
    # The indices in place_strassen are:
    # [u_offset, u_offset+1, u_offset+4, u_offset+5]
    # If u_offset=2: [2, 3, 6, 7]. This matches A[0,2], A[0,3], A[1,2], A[1,3]. Correct.
    
    # B2a is at V rows 2-3, cols 0-1.
    # B[2,0] = 2*7+0 = 14
    # B[2,1] = 2*7+1 = 15
    # B[3,0] = 3*7+0 = 21
    # B[3,1] = 3*7+1 = 22
    # V indices: [14, 15, 21, 22]
    
    # C2a is at W cols 0-1. Offset 0.
    rank_off = place_strassen(U, V, W, 2, [14, 15, 21, 22], 0, rank_off)
    
    # 5. A2 (rows 0-1, cols 2-3) * B2b (rows 2-3, cols 2-3)
    # A2 u_offset = 2.
    # B2b is at V rows 2-3, cols 2-3.
    # B[2,2] = 2*7+2 = 16
    # B[2,3] = 2*7+3 = 17
    # B[3,2] = 3*7+2 = 23
    # B[3,3] = 3*7+3 = 24
    # V indices: [16, 17, 23, 24]
    
    # C2b is at W cols 2-3. Offset 2.
    rank_off = place_strassen(U, V, W, 2, [16, 17, 23, 24], 2, rank_off)
    
    # 6. A2 (rows 0-1, cols 2-3) * B2c (rows 2-3, cols 4-6)
    # A2 u_offset = 2.
    # B2c is at V rows 2-3, cols 4-6. Base row 2, Base col 4.
    # C2c is at W cols 4-6. Offset 4.
    rank_off = place_schoolbook_2x2x3(U, V, W, 2, 2, 4, 4, rank_off)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Build the structured decomposition
    U, V, W = _build_2x4x7_decomposition()
    
    rank = len(U[0])
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
