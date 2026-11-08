import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Strassen coefficients for 2x2x2
    U_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    V_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    W_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    # Build baseline rank 42 solution
    rank_base = 42
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs_222[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U_base[i * m + j][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs_222[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V_base[j * p + k][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs_222[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W_base[k * n + i][col] += coeff
                col += 1
    
    # Try to find rank 40: two 2x2x6 decompositions of rank 20 each
    # The 2x2x6 tensor T_sub has entries T[i,j,k] = 1 if valid index
    n_sub, m_sub, p_sub = 2, 2, 6
    I, J, K = n_sub * m_sub, m_sub * p_sub, p_sub * n_sub  # 4, 12, 12
    
    # Build the tensor as a 3D array
    T_sub = np.zeros((I, J, K), dtype=np.float64)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a_idx = i * m_sub + j
                b_idx = j * p_sub + k
                c_idx = k * n_sub + i
                T_sub[a_idx, b_idx, c_idx] = 1.0
    
    # Strategy: Start from the rank-21 solution (3 Strassen blocks) and try to
    # find a linear dependency to reduce to rank 20.
    # Build the rank-21 solution for 2x2x6
    rank_21 = 21
    U_21 = np.zeros((I, rank_21))
    V_21 = np.zeros((J, rank_21))
    W_21 = np.zeros((K, rank_21))
    
    for b in range(3):
        for s in range(7):
            col_idx = b * 7 + s
            for (i_loc, j_loc), coeff in U_coeffs_222[s].items():
                U_21[i_loc * m_sub + j_loc, col_idx] = coeff
            for (j_loc, k_loc), coeff in V_coeffs_222[s].items():
                V_21[j_loc * p_sub + (b * 2 + k_loc), col_idx] = coeff
            for (i_loc, k_loc), coeff in W_coeffs_222[s].items():
                W_21[(b * 2 + k_loc) * n_sub + i_loc, col_idx] = coeff
    
    # Verify rank 21 solution
    T_recon_21 = np.zeros((I, J, K))
    for s in range(rank_21):
        T_recon_21 += np.einsum('i,j,k->ijk', U_21[:, s], V_21[:, s], W_21[:, s])
    error_21 = np.linalg.norm(T_sub - T_recon_21)
    
    if error_21 > 1e-10:
        # Fallback to rank 42
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Now try to find if we can reduce to rank 20 by checking if the 21 vectors
    # in the "flattened" space are linearly dependent in a way that allows rank reduction.
    # For CP decomposition, rank reduction is hard. Instead, try a different approach:
    # Use alternating least squares starting from the rank-21 solution, but with
    # rank 20, and see if we can converge.
    
    rank_target = 20
    # Initialize rank 20 from rank 21 by dropping the last column (or averaging)
    U_init = U_21[:, :rank_target].copy()
    V_init = V_21[:, :rank_target].copy()
    W_init = W_21[:, :rank_target].copy()
    
    # Better: use SVD-based initialization. Flatten T to (I, J*K)
    T_flat = T_sub.reshape(I, J * K)
    # But CP is not matrix factorization. Use the rank-21 as warm start.
    
    # ALS optimization for rank 20
    U_cur = U_init.copy()
    V_cur = V_init.copy()
    W_cur = W_init.copy()
    
    best_error = np.inf
    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    # Precompute reshaped tensors
    T_ijk = T_sub
    T_jik = T_sub.transpose(1, 0, 2)
    T_kij = T_sub.transpose(2, 0, 1)
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 30:
            break
        
        # Update U: Fix V, W. T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
        # For each (j,k), T[:,j,k] = U * (V[:,s]*W[k,s])_s
        # Let A[j*K+k, s] = V[j,s] * W[k,s]
        # T_flat[i, j*K+k] = sum_s U[i,s] A[j*K+k, s]
        # So T_flat = U * A^T => U = T_flat * pinv(A^T)
        
        A_mat = np.zeros((J * K, rank_target))
        for j in range(J):
            for k in range(K):
                A_mat[j * K + k, :] = V_cur[:, :] * W_cur[k, :]
        
        T_flat_I = T_sub.reshape(I, J * K)
        U_new = T_flat_I @ np.linalg.pinv(A_mat.T)
        
        # Update V: Fix U, W. T[j,i,k] = sum_s V[j,s] U[i,s] W[k,s]
        # B[i*K+k, s] = U[i,s] * W[k,s]
        # T_jik_flat[j, i*K+k] = sum_s V[j,s] B[i*K+k, s]
        # T_jik_flat = V * B^T => V = T_jik_flat * pinv(B^T)
        
        B_mat = np.zeros((I * K, rank_target))
        for i in range(I):
            for k in range(K):
                B_mat[i * K + k, :] = U_new[:, :] * W_cur[k, :]
        
        T_jik_flat = T_jik.reshape(J, I * K)
        V_new = T_jik_flat @ np.linalg.pinv(B_mat.T)
        
        # Update W: Fix U, V. T[k,i,j] = sum_s W[k,s] U[i,s] V[j,s]
        # C[i*J+j, s] = U[i,s] * V[j,s]
        # T_kij_flat[k, i*J+j] = sum_s W[k,s] C[i*J+j, s]
        # T_kij_flat = W * C^T => W = T_kij_flat * pinv(C^T)
        
        C_mat = np.zeros((I * J, rank_target))
        for i in range(I):
            for j in range(J):
                C_mat[i * J + j, :] = U_new[:, :] * V_new[j, :]
        
        T_kij_flat = T_kij.reshape(K, I * J)
        W_new = T_kij_flat @ np.linalg.pinv(C_mat.T)
        
        U_cur, V_cur, W_cur = U_new, V_new, W_new
        
        # Compute error
        T_recon = np.zeros((I, J, K))
        for s in range(rank_target):
            T_recon += np.einsum('i,j,k->ijk', U_cur[:, s], V_cur[:, s], W_cur[:, s])
        error = np.linalg.norm(T_sub - T_recon)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        
        if error < 1e-10:
            break
    
    # If we found a valid rank 20 decomposition, build the full rank 40 solution
    if best_error < 1e-8 and time.time() < deadline - 10:
        rank_full = 40
        U_full = [[0] * rank_full for _ in range(n * m)]
        V_full = [[0] * rank_full for _ in range(m * p)]
        W_full = [[0] * rank_full for _ in range(p * n)]
        
        # Block 1: j in {0, 1}
        for s in range(rank_target):
            for i in range(2):
                for j in range(2):
                    U_full[i * m + j][s] = best_U[i * 2 + j, s]
            for j in range(2):
                for k in range(6):
                    V_full[j * p + k][s] = best_V[j * 6 + k, s]
            for k in range(6):
                for i in range(2):
                    W_full[k * n + i][s] = best_W[k * 2 + i, s]
        
        # Block 2: j in {2, 3}
        col_offset = rank_target
        for s in range(rank_target):
            for i in range(2):
                for j in range(2):
                    j_full = j + 2
                    U_full[i * m + j_full][s + col_offset] = best_U[i * 2 + j, s]
            for j in range(2):
                j_full = j + 2
                for k in range(6):
                    V_full[j_full * p + k][s + col_offset] = best_V[j * 6 + k, s]
            for k in range(6):
                for i in range(2):
                    W_full[k * n + i][s + col_offset] = best_W[k * 2 + i, s]
        
        # Verify the full solution
        U_np = np.array(U_full, dtype=np.float64)
        V_np = np.array(V_full, dtype=np.float64)
        W_np = np.array(W_full, dtype=np.float64)
        
        # Check: C[i,k] = sum_j A[i,j] B[j,k]
        # Using the decomposition: C[i,k] = sum_s (sum_j U[i*m+j, s] * A[i,j]) * (sum_j V[j*p+k, s] * B[j,k]) * W[k*n+i, s]
        # Actually the decomposition is: T[i,j,k] = sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s]
        # So C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_s U[i*m+j,s] V[j*p+k,s] W[k*n+i,s] * (A[i,j]*B[j,k] normalized?)
        # The tensor T[i,j,k] = 1 means the coefficient is 1, so:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_s (sum_j U[i*m+j,s] A[i,j]) (sum_j V[j*p+k,s] B[j,k]) W[k*n+i,s]
        # This is NOT how it works. The correct interpretation:
        # The tensor T_{ijk} represents the coefficient such that:
        # C_{ik} = sum_{j} A_{ij} B_{jk} = sum_s (sum_j U_{i,j,s} A_{ij}) (sum_j V_{j,k,s} B_{jk}) W_{k,i,s}
        # Wait, that's not right either. Let me think again.
        #
        # Standard tensor decomposition for matrix multiplication:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # = sum_s (sum_j U_{ij}^{(s)} A_{ij}) (sum_j V_{jk}^{(s)} B_{jk}) W_{ik}^{(s)}
        #
        # So U has shape (n*m, rank) where U[i*m+j, s] is the coefficient for A[i,j] in term s
        # V has shape (m*p, rank) where V[j*p+k, s] is the coefficient for B[j,k] in term s
        # W has shape (p*n, rank) where W[k*n+i, s] is the coefficient for C[i,k] in term s
        #
        # And the identity is: sum_j A_{ij} B_{jk} = sum_s (sum_j U[i*m+j,s] A_{ij}) (sum_j V[j*p+k,s] B_{jk}) W[k*n+i,s]
        #
        # This means: T[i,j,k] (the tensor for the multiplication) satisfies:
        # T[i,j,k] = sum_s U[i*m+j,s] V[j*p+k,s] W[k*n+i,s]
        #
        # So we need to verify: for all i,j,k, sum_s U[i*m+j,s] V[j*p+k,s] W[k*n+i,s] = 1
        # (since the matrix multiplication tensor has 1s at valid positions)
        
        # Build the full tensor and verify
        T_full = np.zeros((n * m, m * p, p * n))
        for s in range(rank_full):
            T_full += np.einsum('i,j,k->ijk', U_np[:, s], V_np[:, s], W_np[:, s])
        
        # The expected tensor has T_full[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        # And 0 elsewhere
        error_full = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = T_full[i * m + j, j * p + k, k * n + i]
                    error_full += (expected - actual) ** 2
        # Also check that other entries are 0
        for idx in range(n * m):
            for idx2 in range(m * p):
                for idx3 in range(p * n):
                    i = idx // m
                    j = idx % m
                    j2 = idx2 // p
                    k2 = idx2 % p
                    k = idx3 // n
                    i2 = idx3 % n
                    if not (j == j2 and i == i2):
                        error_full += T_full[idx, idx2, idx3] ** 2
        
        if error_full < 1e-6:
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to rank 42
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
