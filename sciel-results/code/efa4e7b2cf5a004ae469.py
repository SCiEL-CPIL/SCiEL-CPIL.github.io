"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build rank-54 Strassen decomposition
    rank_base = 54
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] -= 1
        col += 1
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+k_base][col] += 1
        V[j_base*p+k_base][col] -= 1
        W[k_base*n+i_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        U[i_base*m+j_base][col] += 1
        U[i_base*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        U[i_base*m+j_base][col] -= 1
        V[j_base*p+k_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        U[i_base*m+(j_base+1)][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] -= 1
        V[(j_base+1)*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    if col != rank_base:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][i*m*p + j*p + k] = 1
                    V[j*p+k][i*m*p + j*p + k] = 1
                    W[k*n+i][i*m*p + j*p + k] = 1
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    # SVD-based rank reduction: try to eliminate linearly dependent terms
    # using the null space structure of the decomposition
    if time.time() < deadline - 30.0:
        # Attempt iterative null-space elimination
        U_cur = U_arr.copy()
        V_cur = V_arr.copy()
        W_cur = W_arr.copy()
        current_rank = rank_base
        tol = 1e-10
        
        for iteration in range(20):
            if time.time() > deadline - 15.0:
                break
            
            # Check which columns are linearly dependent
            # Use the fact that if column r is a linear combination of others,
            # we can express it and eliminate
            
            # Find linearly dependent columns via SVD of U
            U_svd = U_cur[:, :current_rank]
            V_svd = V_cur[:, :current_rank]
            W_svd = W_cur[:, :current_rank]
            
            # Check rank of U
            U_rank = np.linalg.matrix_rank(U_svd, tol=tol)
            V_rank = np.linalg.matrix_rank(V_svd, tol=tol)
            W_rank = np.linalg.matrix_rank(W_svd, tol=tol)
            
            if U_rank < current_rank or V_rank < current_rank or W_rank < current_rank:
                # There's a linear dependence; try to eliminate one term
                # Find which column to eliminate
                # Use null space of U
                if U_rank < current_rank:
                    U_t, _, U_vt = np.linalg.svd(U_svd, full_matrices=True)
                    # Null space vectors are in rows of U_vt beyond rank
                    null_U = U_vt[U_rank:, :]
                elif V_rank < current_rank:
                    V_t, _, V_vt = np.linalg.svd(V_svd, full_matrices=True)
                    null_U = V_vt[V_rank:, :]
                else:
                    W_t, _, W_vt = np.linalg.svd(W_svd, full_matrices=True)
                    null_U = W_vt[W_rank:, :]
                
                # For each null vector, try to eliminate the corresponding column
                eliminated = False
                for null_vec in null_U:
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Find the column with largest coefficient in this null vector
                    coeffs = null_vec
                    max_idx = np.argmax(np.abs(coeffs))
                    max_coeff = coeffs[max_idx]
                    
                    if abs(max_coeff) < tol:
                        continue
                    
                    # Try to eliminate column max_idx by adding/subtracting
                    # scaled versions of other columns
                    new_U = U_cur[:, :current_rank].copy()
                    new_V = V_cur[:, :current_rank].copy()
                    new_W = W_cur[:, :current_rank].copy()
                    
                    # For each other column j, add (coeffs[j]/max_coeff) * column_j to column_max_idx
                    for j in range(current_rank):
                        if j == max_idx:
                            continue
                        scale = coeffs[j] / max_coeff
                        if abs(scale) > tol:
                            new_U[:, max_idx] += scale * new_U[:, j]
                            new_V[:, max_idx] += scale * new_V[:, j]
                            new_W[:, max_idx] += scale * new_W[:, j]
                    
                    # Check if this maintains the decomposition
                    # The key insight: if the null vector is in the null space of the
                    # "decomposition operator", then this transformation preserves the tensor
                    test_U = np.concatenate([new_U[:, :max_idx], new_U[:, max_idx+1:]], axis=1)
                    test_V = np.concatenate([new_V[:, :max_idx], new_V[:, max_idx+1:]], axis=1)
                    test_W = np.concatenate([new_W[:, :max_idx], new_W[:, max_idx+1:]], axis=1)
                    
                    if test_U.shape[1] == current_rank - 1:
                        T_test = np.einsum('ar,br,cr->abc', test_U, test_V, test_W)
                        if np.allclose(T_test, T, atol=1e-8):
                            # Successfully eliminated one term!
                            U_cur = test_U
                            V_cur = test_V
                            W_cur = test_W
                            current_rank -= 1
                            eliminated = True
                            break
                    
                    if eliminated:
                        break
            
            if not eliminated:
                break
        
        # If we reduced the rank, use the reduced decomposition
        if current_rank < rank_base:
            U_final = U_cur.tolist()
            V_final = V_cur.tolist()
            W_final = W_cur.tolist()
            
            # Final verification
            U_chk = np.array(U_final, dtype=np.float64)
            V_chk = np.array(V_final, dtype=np.float64)
            W_chk = np.array(W_final, dtype=np.float64)
            T_chk = np.einsum('ar,br,cr->abc', U_chk, V_chk, W_chk)
            
            if np.allclose(T_chk, T, atol=1e-8):
                return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: return the original rank-54 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
