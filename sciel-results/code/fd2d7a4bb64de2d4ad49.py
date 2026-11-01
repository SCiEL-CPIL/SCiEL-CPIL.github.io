"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    # Strategy:
    # 1. Try to find a rank-11 decomposition for the 2x3x2 sub-tensor.
    #    If found, tile it 4 times to get a rank-44 decomposition for 2x3x8.
    # 2. If not found or if time runs out, fall back to schoolbook rank 48.

    def find_decomposition_2x3x2(max_rank=12, time_limit=50.0):
        """
        Attempt to find a rank-r decomposition of the 2x3x2 matrix multiplication tensor.
        Returns (r, U, V, W) if found, else None.
        U: 6xr, V: 6xr, W: 4xr
        """
        n2, m2, p2 = 2, 3, 2
        dim_u = n2 * m2  # 6
        dim_v = m2 * p2  # 6
        dim_w = p2 * n2  # 4

        # Build the target tensor T
        T = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T[i*m2+j, j*p2+k, k*n2+i] = 1.0

        start_time = time.time()
        
        # We suspect rank 11 might be possible. Let's try r=11 first.
        # If r=11 fails, we might try r=10, 9... but 11 is the most likely improvement over 12.
        for r in [11, 10, 12]: 
            if time.time() > start_time + time_limit:
                break
            
            # Use random restarts with gradient descent
            found = False
            for restart in range(20):
                if time.time() > start_time + time_limit:
                    break
                
                np.random.seed(1000 + r * 100 + restart)
                U = np.random.randn(dim_u, r) * 0.5
                V = np.random.randn(dim_v, r) * 0.5
                W = np.random.randn(dim_w, r) * 0.5

                lr = 0.02
                best_err = float('inf')
                
                for it in range(2000):
                    if time.time() > start_time + time_limit:
                        break
                    
                    # Compute reconstruction
                    recon = np.zeros((dim_u, dim_v, dim_w))
                    for c in range(r):
                        recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                    
                    err = T - recon
                    err_norm = np.linalg.norm(err)
                    
                    if err_norm < 1e-9:
                        found = True
                        break
                        
                    if err_norm < best_err:
                        best_err = err_norm
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                    # Gradients
                    dU = np.einsum('ijk,jc,kc->ic', err, V, W)
                    dV = np.einsum('ijk,ic,kc->jc', err, U, W)
                    dW = np.einsum('ijk,ic,jc->kc', err, U, V)
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW

                if found:
                    # Verify exactness with the best found or current
                    recon_final = np.zeros((dim_u, dim_v, dim_w))
                    for c in range(r):
                        recon_final += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                    if np.linalg.norm(T - recon_final) < 1e-9:
                        # Try to round to half-integers
                        U_round = np.round(2*U) / 2.0
                        V_round = np.round(2*V) / 2.0
                        W_round = np.round(2*W) / 2.0
                        
                        recon_r = np.zeros((dim_u, dim_v, dim_w))
                        for c in range(r):
                            recon_r += np.einsum('i,j,k->ijk', U_round[:,c], V_round[:,c], W_round[:,c])
                        if np.linalg.norm(T - recon_r) < 1e-9:
                            return r, U_round, V_round, W_round
                        
                        # If rounding fails, try integers
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        recon_i = np.zeros((dim_u, dim_v, dim_w))
                        for c in range(r):
                            recon_i += np.einsum('i,j,k->ijk', U_int[:,c], V_int[:,c], W_int[:,c])
                        if np.linalg.norm(T - recon_i) < 1e-9:
                            return r, U_int, V_int, W_int

        return None

    # Try to find the sub-decomposition
    result_232 = find_decomposition_2x3x2(max_rank=12, time_limit=60.0)

    if result_232 is not None:
        r_sub, U_sub, V_sub, W_sub = result_232
        # Build the full 2x3x8 decomposition by tiling
        rank_total = 4 * r_sub
        U_full = np.zeros((6, rank_total))
        V_full = np.zeros((24, rank_total))
        W_full = np.zeros((16, rank_total))

        for g in range(4):
            col_offset = g * r_sub
            for c in range(r_sub):
                # U is the same for all groups (rows of A, cols of A/rows of B)
                U_full[:, col_offset + c] = U_sub[:, c]
                # V: j*8 + (g*2 + k_sub)
                for j in range(3):
                    for k_sub in range(2):
                        V_full[j*8 + g*2 + k_sub, col_offset + c] = V_sub[j*2 + k_sub, c]
                # W: (g*2 + k_sub)*2 + i
                for k_sub in range(2):
                    for i in range(2):
                        W_full[(g*2 + k_sub)*2 + i, col_offset + c] = W_sub[k_sub*2 + i, c]

        # Verify the full decomposition
        T_full = np.zeros((6, 24, 16))
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    T_full[i*3+j, j*8+k, k*2+i] = 1.0

        recon_full = np.zeros((6, 24, 16))
        for c in range(rank_total):
            recon_full += np.einsum('i,j,k->ijk', U_full[:,c], V_full[:,c], W_full[:,c])
        
        if np.linalg.norm(T_full - recon_full) < 1e-9:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: schoolbook decomposition (Rank 48)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
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
