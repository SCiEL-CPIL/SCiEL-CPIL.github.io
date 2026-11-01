import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    # Build rank-36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1
    
    # Block 1: A_L * B_11
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R * B_21
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L * B_12
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R * B_22
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L * B_15 (2x2 x 2x1)
    for (ai, aj), (bi, bk), (ci) in [((0,0),(0,4),0), ((0,1),(1,4),0), ((1,0),(0,4),1), ((1,1),(1,4),1)]:
        U[u_idx(ai, aj)][col] = 1
        V[v_idx(bi, bk)][col] = 1
        W[w_idx(4, ci)][col] = 1
        col += 1
    
    # Block 6: A_R * B_25 (2x2 x 2x1)
    for (ai, aj), (bi, bk), (ci) in [((0,2),(2,4),0), ((0,3),(3,4),0), ((1,2),(2,4),1), ((1,3),(3,4),1)]:
        U[u_idx(ai, aj)][col] = 1
        V[v_idx(bi, bk)][col] = 1
        W[w_idx(4, ci)][col] = 1
        col += 1
    
    # Convert to numpy for analysis
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the decomposition
    # Tensor T[i,j,k] should equal sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (i_row*m + i_col), j indexes B (j_row*p + j_col), k indexes C (k_row*n + k_col)
    # For matrix mult: C[k_row, k_col] = sum_m A[i_row, m] * B[m, k_col]
    # In our indexing: A index = i_row*m + m, B index = m*p + k_col, C index = k_col*n + i_row
    
    # Build the true tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Build decomposed tensor
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U_np[:,r], V_np[:,r], W_np[:,r])
    
    # Check if they match
    if not np.allclose(T_true, T_decomp, atol=1e-10):
        # Fallback: return the rank-36 decomposition as-is
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank by checking linear dependencies
    # The 36 rank-1 tensors form a set. If any is a linear combination of others, we can remove it.
    # Represent each rank-1 tensor as a vector in R^(n*m * m*p * p*n)
    flat_size = n*m * m*p * p*n
    tensors = np.zeros((rank, flat_size))
    for r in range(rank):
        tensors[r] = np.einsum('i,j,k->ijk', U_np[:,r], V_np[:,r], W_np[:,r]).flatten()
    
    # Check rank of the tensor matrix
    # If rank < 36, we can reduce
    tensor_rank = np.linalg.matrix_rank(tensors, tol=1e-10)
    
    if tensor_rank < rank:
        # Find which columns can be removed
        # Use QR decomposition to find a basis
        Q, R = np.linalg.qr(tensors.T)  # (flat_size, rank) and (rank, rank)
        # R is upper triangular; find which diagonal entries are non-zero
        diag = np.abs(np.diag(R))
        pivot_cols = [i for i in range(rank) if diag[i] > 1e-10]
        
        # Keep only pivot columns
        new_rank = len(pivot_cols)
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        # Rebuild with only pivot columns
        U_final = [[0]*new_rank for _ in range(n*m)]
        V_final = [[0]*new_rank for _ in range(m*p)]
        W_final = [[0]*new_rank for _ in range(p*n)]
        
        for new_idx, old_idx in enumerate(pivot_cols):
            for i in range(n*m):
                U_final[i][new_idx] = U[old_idx][i] if False else U[i][old_idx]
            for j in range(m*p):
                V_final[j][new_idx] = V[j][old_idx]
            for k in range(p*n):
                W_final[k][new_idx] = W[k][old_idx]
        
        # Verify the reduced decomposition
        U_np_new = np.array(U_final, dtype=float)
        V_np_new = np.array(V_final, dtype=float)
        W_np_new = np.array(W_final, dtype=float)
        
        T_decomp_new = np.zeros((n*m, m*p, p*n))
        for r in range(new_rank):
            T_decomp_new += np.einsum('i,j,k->ijk', U_np_new[:,r], V_np_new[:,r], W_np_new[:,r])
        
        if np.allclose(T_true, T_decomp_new, atol=1e-10):
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Return the original rank-36 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
