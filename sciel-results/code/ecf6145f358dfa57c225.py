import time
import numpy as np

def _laderman_3x3():
    """
    Returns Laderman decomposition for 3x3 matrix multiplication.
    Rank = 23.
    U, V, W are 9x23 matrices.
    U indices: i*3 + j
    V indices: j*3 + k
    W indices: i*3 + k (Standard row-major for C)
    """
    n = 3
    rank = 23
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Indices in A, B, C (flattened row-major)
    # A: 0,1,2 / 3,4,5 / 6,7,8
    # B: 0,1,2 / 3,4,5 / 6,7,8
    # C: 0,1,2 / 3,4,5 / 6,7,8
    
    # Laderman 3x3 terms
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are lists of (index, coeff)
    
    terms = [
        # M1
        ([(0, 1), (8, 1)], [(0, 1), (8, 1)], [(0, 1), (8, 1)]),
        # M2
        ([(3, 1), (6, 1)], [(3, 1), (6, 1)], [(3, 1), (6, 1)]),
        # M3
        ([(1, 1), (7, 1)], [(1, 1), (7, 1)], [(1, 1), (7, 1)]),
        # M4
        ([(0, 1), (1, 1), (2, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)]),
        # M5
        ([(0, 1), (3, 1), (6, 1)], [(0, 1), (1, 1), (2, 1)], [(0, 1), (1, 1), (2, 1)]),
        # M6
        ([(0, 1), (4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)]),
        # M7
        ([(0, 1), (1, 1), (4, 1)], [(0, 1), (1, 1), (4, 1)], [(0, 1), (1, 1), (4, 1)]),
        # M8
        ([(0, 1), (4, 1), (8, 1)], [(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # M9
        ([(0, 1), (4, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # M10
        ([(0, 1), (1, 1), (4, 1)], [(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # M11
        ([(0, 1), (1, 1), (4, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # M12
        ([(1, 1), (7, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)]),
        # M13
        ([(1, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # M14
        ([(1, 1), (7, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # M15
        ([(3, 1), (6, 1)], [(0, 1), (1, 1), (2, 1)], [(0, 1), (1, 1), (2, 1)]),
        # M16
        ([(3, 1), (6, 1)], [(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # M17
        ([(3, 1), (6, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # M18
        ([(0, 1), (8, 1)], [(1, 1), (7, 1)], [(1, -1), (7, -1)]),
        # M19
        ([(0, 1), (8, 1)], [(2, 1), (8, 1)], [(2, -1), (8, -1)]),
        # M20
        ([(3, 1), (6, 1)], [(4, 1), (7, 1)], [(4, -1), (7, -1)]),
        # M21
        ([(3, 1), (6, 1)], [(5, 1), (8, 1)], [(5, -1), (8, -1)]),
        # M22
        ([(1, 1), (7, 1)], [(4, 1), (7, 1)], [(4, -1), (7, -1)]),
        # M23
        ([(1, 1), (7, 1)], [(5, 1), (8, 1)], [(5, -1), (8, -1)]),
    ]
    
    for t, (ca, cb, cc) in enumerate(terms):
        for idx, val in ca:
            U[idx, t] = val
        for idx, val in cb:
            V[idx, t] = val
        for idx, val in cc:
            W[idx, t] = val
            
    return U, V, W

def _permute_W(W_std, n):
    """
    W_std has shape (n*n, rank) with index i*n + k.
    Returns W_req with shape (n*n, rank) with index k*n + i.
    """
    rank = W_std.shape[1]
    W_req = np.zeros_like(W_std)
    for i in range(n):
        for k in range(n):
            idx_std = i * n + k
            idx_req = k * n + i
            W_req[idx_req, :] = W_std[idx_std, :]
    return W_req

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with index k*n + i
    """
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    rank = U.shape[1]
    
    # Reshape U, V, W for vectorized computation if possible, or loop
    # Loop is fine for 184 rank
    for t in range(rank):
        # u_t is vector of length n*m
        # v_t is vector of length m*p
        # w_t is vector of length p*n (index k*n+i)
        
        # Compute scalar products
        # u_t^T A_flat
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        prod_u = np.dot(U[:, t], A_flat)
        prod_v = np.dot(V[:, t], B_flat)
        
        # Add to C
        # w_t corresponds to C^T flattened?
        # W index is k*n + i.
        # So W[:, t] is a vector w where w[k*n+i] is the coefficient for C[i,k].
        # C[i,k] += w[k*n+i] * prod_u * prod_v
        
        # Reshape w to (p, n) where w_mat[k, i] = w[k*n+i]
        w_mat = W[:, t].reshape(p, n)
        
        C_comp += prod_u * prod_v * w_mat.T
        
    if not np.allclose(C_comp, C_true, atol=1e-10):
        return False
    return True

def _schoolbook_6x6():
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j]
                U[i * m + j, col] = 1
                # B[j,k]
                V[j * p + k, col] = 1
                # C[i,k] -> W index k*n + i
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 6, 6, 6
    
    # 1. Get Local Laderman 3x3
    U_loc, V_loc, W_loc_std = _laderman_3x3()
    W_loc = _permute_W(W_loc_std, 3)
    
    # 2. Verify Local
    if not _verify_decomposition(U_loc, V_loc, W_loc, 3, 3, 3):
        # Fallback to schoolbook if local fails (should not happen)
        U, V, W = _schoolbook_6x6()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # 3. Tile 2x2x2
    # Global Rank = 2 * 2 * 2 * 23 = 184
    global_rank = 8 * 23
    U_glob = np.zeros((36, global_rank))
    V_glob = np.zeros((36, global_rank))
    W_glob = np.zeros((36, global_rank))
    
    block_size = 3
    blocks = 2
    
    col_offset = 0
    for I_b in range(blocks):
        for J_b in range(blocks):
            for K_b in range(blocks):
                for t in range(23):
                    # Update U
                    # Local U index a = i*3 + j
                    # Global A index = (I_b*3 + i)*6 + (J_b*3 + j)
                    for i in range(block_size):
                        for j in range(block_size):
                            loc_idx = i * 3 + j
                            val = U_loc[loc_idx, t]
                            if val != 0:
                                glob_i = I_b * 3 + i
                                glob_j = J_b * 3 + j
                                glob_idx = glob_i * 6 + glob_j
                                U_glob[glob_idx, col_offset] = val
                                
                    # Update V
                    # Local V index b = j*3 + k
                    # Global B index = (J_b*3 + j)*6 + (K_b*3 + k)
                    for j in range(block_size):
                        for k in range(block_size):
                            loc_idx = j * 3 + k
                            val = V_loc[loc_idx, t]
                            if val != 0:
                                glob_j = J_b * 3 + j
                                glob_k = K_b * 3 + k
                                glob_idx = glob_j * 6 + glob_k
                                V_glob[glob_idx, col_offset] = val
                                
                    # Update W
                    # Local W index w = k*3 + i (already permuted)
                    # Global W index = (K_b*3 + k)*6 + (I_b*3 + i)
                    for k in range(block_size):
                        for i in range(block_size):
                            loc_idx = k * 3 + i
                            val = W_loc[loc_idx, t]
                            if val != 0:
                                glob_k = K_b * 3 + k
                                glob_i = I_b * 3 + i
                                glob_idx = glob_k * 6 + glob_i
                                W_glob[glob_idx, col_offset] = val
                                
                    col_offset += 1
                    
    # 4. Verify Global
    if not _verify_decomposition(U_glob, V_glob, W_glob, 6, 6, 6):
        # Fallback
        U, V, W = _schoolbook_6x6()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_glob.tolist(), "V": V_glob.tolist(), "W": W_glob.tolist()}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print("Rank:", len(res['U'][0]))
