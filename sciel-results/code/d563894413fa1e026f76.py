"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Strategy: Use Strassen's algorithm for 2x2 blocks of B.
    # Split B (2x9) into four 2x2 blocks and one 2x1 block.
    # Rank for 2x2x2 is 7. Rank for 2x2x1 is 4.
    # Total rank = 4 * 7 + 4 = 32.
    
    rank = 32
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    col = 0
    
    # Strassen terms for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    
    # C00 = m1 - m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 - m3 + m6
    
    strassen_terms = [
        # U, V, W
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # m3
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]), # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),  # m6
        ([0, 1, -1, 0], [0, 1, 0, 1], [1, 0, 0, -1])  # m7
    ]
    
    # Process 4 blocks of 2 columns
    for block_idx in range(4):
        # Columns in B: 2*block_idx, 2*block_idx + 1
        col_offset = 2 * block_idx
        
        for term in strassen_terms:
            u_vec, v_vec, w_vec = term
            
            # U is always the same 4 entries
            for i in range(4):
                U[i, col] = u_vec[i]
                
            # V maps to B entries
            # B indices:
            # e = B[0, col_offset]   -> index 0*9 + col_offset
            # f = B[0, col_offset+1] -> index 0*9 + col_offset + 1
            # g = B[1, col_offset]   -> index 1*9 + col_offset
            # h = B[1, col_offset+1] -> index 1*9 + col_offset + 1
            
            v_indices = [col_offset, col_offset + 1, 9 + col_offset, 9 + col_offset + 1]
            for i in range(4):
                V[v_indices[i], col] = v_vec[i]
                
            # W maps to C entries
            # C indices:
            # C00 = C[col_offset, 0] -> index col_offset*2 + 0
            # C01 = C[col_offset+1, 0] -> index (col_offset+1)*2 + 0
            # C10 = C[col_offset, 1] -> index col_offset*2 + 1
            # C11 = C[col_offset+1, 1] -> index (col_offset+1)*2 + 1
            
            w_indices = [
                col_offset * 2, 
                (col_offset + 1) * 2, 
                col_offset * 2 + 1, 
                (col_offset + 1) * 2 + 1
            ]
            for i in range(4):
                W[w_indices[i], col] = w_vec[i]
                
            col += 1
            
    # Process last block of 1 column (column 8)
    # Schoolbook for 2x2x1
    # C00 = a*e + b*g
    # C10 = c*e + d*g
    # Note: C is 2x9, so last column is index 8.
    # C[8, 0] = C00, C[8, 1] = C10
    
    last_col = 8
    
    # Term 1: a * B[0, 8] -> C[8, 0]
    U[0, col] = 1
    V[0*9 + last_col, col] = 1
    W[last_col*2 + 0, col] = 1
    col += 1
    
    # Term 2: b * B[1, 8] -> C[8, 0]
    U[1, col] = 1
    V[1*9 + last_col, col] = 1
    W[last_col*2 + 0, col] = 1
    col += 1
    
    # Term 3: c * B[0, 8] -> C[8, 1]
    U[2, col] = 1
    V[0*9 + last_col, col] = 1
    W[last_col*2 + 1, col] = 1
    col += 1
    
    # Term 4: d * B[1, 8] -> C[8, 1]
    U[3, col] = 1
    V[1*9 + last_col, col] = 1
    W[last_col*2 + 1, col] = 1
    col += 1
    
    # Verification
    # Check if U @ diag(V) @ W^T reconstructs the tensor?
    # Tensor T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # We want T to be the matrix multiplication tensor.
    # A is 2x2, B is 2x9. C is 2x9.
    # T[A_idx, B_idx, C_idx] should be 1 if C_idx corresponds to A_idx * B_idx.
    
    # Let's do a numerical check with random matrices
    np.random.seed(42)
    A_test = np.random.rand(2, 2)
    B_test = np.random.rand(2, 9)
    C_test = A_test @ B_test
    
    # Reconstruct C using factors
    # C_recon[i, k] = sum_j A[i, j] B[j, k]
    # Using factors:
    # C_recon[i, k] = sum_r (sum_j U[i*m+j, r] A[i*m+j]) * V[j*p+k, r] * W[k*n+i, r] ?
    # No, the decomposition is:
    # C[k, i] = sum_r U[i*m+j, r] ... wait.
    # Standard: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # Here:
    # U is (4, R). Index i in U corresponds to A index (i_row, i_col).
    # V is (18, R). Index j in V corresponds to B index (j_row, j_col).
    # W is (18, R). Index k in W corresponds to C index (k_col, k_row).
    
    # C_recon[k_row, k_col] = sum_r U[?, r] V[?, r] W[k_col*n + k_row, r]
    # Actually, the contraction is:
    # C[k_row, k_col] = sum_{a_idx, b_idx} A[a_idx] B[b_idx] T[a_idx, b_idx, k_row*n+k_col]
    # T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] V[b_idx, r] W[c_idx, r]
    
    C_recon = np.zeros((2, 9))
    for a_idx in range(4):
        for b_idx in range(18):
            # T[a_idx, b_idx, c_idx]
            # c_idx is determined by a_idx and b_idx? No.
            # We sum over r.
            # For each c_idx, T is sum_r U[a,r] V[b,r] W[c,r]
            # We want to compute C = A @ B.
            # C[i, k] = sum_j A[i, j] B[j, k]
            # A[i, j] is A_flat[i*2+j]
            # B[j, k] is B_flat[j*9+k]
            # C[i, k] is C_flat[k*2+i]
            
            # Let's just compute the full tensor T and check.
            pass
            
    # Efficient check:
    # C_recon = (A_flat @ U) * (V @ W.T) ? No.
    # C_recon[c_idx] = sum_r W[c_idx, r] * (sum_a U[a, r] A[a]) * (sum_b V[b, r] B[b]) ? No.
    # The rank-1 decomposition is sum_r (U[:, r] outer V[:, r] outer W[:, r]).
    # So T = sum_r np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    # We want T[a, b, c] to be 1 if c corresponds to a*b.
    
    # Let's just pick a few random A, B and check C.
    # C_recon[i, k] = sum_j A[i, j] B[j, k]
    # Using factors:
    # C_recon[i, k] = sum_r (sum_j U[i*2+j, r] A[i*2+j]) * (sum_k V[j*9+k, r] B[j*9+k]) * W[k*2+i, r] ??
    # No, the indices are coupled.
    
    # Correct contraction:
    # C[i, k] = sum_{j} A[i, j] B[j, k]
    # This is equivalent to:
    # C[i, k] = sum_r (sum_j U[i*2+j, r] A[i*2+j]) * V[j*9+k, r] * W[k*2+i, r] is WRONG because j is summed in U and V separately?
    # No, the term is U[a, r] V[b, r] W[c, r].
    # C[c] = sum_{a, b} A[a] B[b] T[a, b, c]
    # C[c] = sum_{a, b} A[a] B[b] sum_r U[a, r] V[b, r] W[c, r]
    # C[c] = sum_r W[c, r] (sum_a U[a, r] A[a]) (sum_b V[b, r] B[b])
    
    # Let's compute this.
    U_np = U
    V_np = V
    W_np = W
    
    A_flat = A_test.flatten() # size 4
    B_flat = B_test.flatten() # size 18
    
    # u_proj[r] = sum_a U[a, r] A[a]
    u_proj = U_np.T @ A_flat # size R
    # v_proj[r] = sum_b V[b, r] B[b]
    v_proj = V_np.T @ B_flat # size R
    
    # c_proj[c] = sum_r W[c, r] u_proj[r] v_proj[r]
    # c_proj = W @ (u_proj * v_proj)
    c_flat = W_np @ (u_proj * v_proj) # size 18
    
    C_recon = c_flat.reshape(9, 2).T # C is 2x9, c_flat is k*2+i -> reshape to (9,2) then T to (2,9)
    
    if not np.allclose(C_recon, C_test, atol=1e-5):
        # Fallback to schoolbook
        rank = 36
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((18, rank), dtype=float)
        W = np.zeros((18, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
