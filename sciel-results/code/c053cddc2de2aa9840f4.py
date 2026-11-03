import numpy as np
import time

def construct():
    start_time = time.time()
    
    # 2x2 Strassen Matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker products for 4x4
    # Rank = 7 * 7 = 49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation to map Kronecker indices to Linear Row-Major indices
    # Kronecker index K corresponds to (block_row, block_col, inner_row, inner_col)
    # For U (A matrix):
    # A is 4x4. We split into 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Linear index S = r * 4 + c.
    # r = 2 * r_block + r_inner
    # c = 2 * c_block + c_inner
    # Kronecker index K from np.kron(U2, U2):
    # np.kron(A, B) index (i, j) maps to A[i//p] B[i%p, j//q] B[i%p, j%q]?
    # Actually, np.kron(A, B) has shape (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]).
    # Row index R = i_A * p + i_B.
    # Here p=2. R = i_A * 2 + i_B.
    # So R corresponds to (r_block, r_inner) where r_block = R//2, r_inner = R%2.
    # Similarly for columns (rank dimension), but we only care about rows for U, V, W.
    # For U, row index R in U4 corresponds to (r_block, r_inner) in the 2x2 block structure of A.
    # We want to map this to linear index S = r * 4 + c.
    # Wait, U is for A. A has rows r and cols c.
    # The Kronecker product U4 = kron(U2, U2) has rows indexed by (r_block, r_inner) AND (c_block, c_inner)?
    # No. U2 is 4x7. The 4 rows correspond to the 4 entries of the 2x2 matrix A.
    # Entry (i, j) of A has index i*2+j in U2.
    # So U2 row index L2 = i*2 + j.
    # U4 = kron(U2, U2). Row index L4 = L2_outer * 4 + L2_inner?
    # np.kron(A, B) row index is i_A * B.shape[0] + i_B.
    # So L4 = L2_outer * 4 + L2_inner.
    # L2_outer corresponds to the "outer" 2x2 block indices of A.
    # L2_inner corresponds to the "inner" 2x2 block indices of A.
    # Let L2_outer = r_out * 2 + c_out.
    # Let L2_inner = r_in * 2 + c_in.
    # Then L4 = (r_out * 2 + c_out) * 4 + (r_in * 2 + c_in).
    # We want to map L4 to S = r * 4 + c.
    # r = r_out * 2 + r_in.
    # c = c_out * 2 + c_in.
    # S = (r_out * 2 + r_in) * 4 + (c_out * 2 + c_in)
    #   = r_out * 8 + r_in * 4 + c_out * 2 + c_in.
    
    # So we need a permutation perm such that:
    # U4_fixed[S, :] = U4[L4, :]
    # where L4 is derived from S.
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out = r // 2
        r_in = r % 2
        c_out = c // 2
        c_in = c % 2
        
        L2_out = r_out * 2 + c_out
        L2_in = r_in * 2 + c_in
        
        L4 = L2_out * 4 + L2_in
        perm[S] = L4
        
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Fallback if timeout (unlikely)
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
