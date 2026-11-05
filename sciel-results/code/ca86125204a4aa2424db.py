"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    deadline = 120.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 3, 3, 6
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def try_rank_als(r, max_iters=500):
        if time.time() > deadline - 5:
            return None
        
        # Use random restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update W
                # W_new[k, i, r] = sum_j U[i, j, r] * V[j, k, r]
                # einsum: 'ijr,jkr->ikr' gives [i, k, r]
                # We need [k, i, r]
                tmp = np.einsum('ijr,jkr->ikr', U_r, V_r)
                norms = np.sum(tmp**2, axis=2, keepdims=True)
                norms[norms < 1e-12] = 1.0
                W_r = tmp / norms
                W_r = W_r.transpose(1, 0, 2) # [i, k, r] -> [k, i, r]
                
                # Update U
                # U_new[i, j, r] = sum_k V[j, k, r] * W[k, i, r]
                # einsum: 'jkr,kir->ijr'
                tmp = np.einsum('jkr,kir->ijr', V_r, W_r)
                norms = np.sum(tmp**2, axis=2, keepdims=True)
                norms[norms < 1e-12] = 1.0
                U_r = tmp / norms
                
                # Update V
                # V_new[j, k, r] = sum_i U[i, j, r] * W[k, i, r]
                # einsum: 'ijr,kir->jkr'
                tmp = np.einsum('ijr,kir->jkr', U_r, W_r)
                norms = np.sum(tmp**2, axis=2, keepdims=True)
                norms[norms < 1e-12] = 1.0
                V_r = tmp / norms
                
                # Check error
                if it % 10 == 0:
                    # Compute residuals
                    # Res[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                    # einsum: 'ijr,jkr,kir->ijk'
                    prod = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    err = np.max(np.abs(prod - 1.0))
                    if err < 1e-10:
                        # Snap to rational
                        U_flat = U_r.reshape(n*m, r)
                        V_flat = V_r.reshape(m*p, r)
                        W_flat = W_r.reshape(p*n, r)
                        
                        # Try rounding to half-integers
                        U_test = np.round(U_flat * 2) / 2
                        V_test = np.round(V_flat * 2) / 2
                        W_test = np.round(W_flat * 2) / 2
                        if verify(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                            return (U_test.tolist(), V_test.tolist(), W_test.tolist())
                        
                        # Try integer rounding
                        U_test = np.round(U_flat)
                        V_test = np.round(V_flat)
                        W_test = np.round(W_flat)
                        if verify(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                            return (U_test.tolist(), V_test.tolist(), W_test.tolist())
                        
                        # If not exact rational, but very close, maybe we need higher precision or different snap?
                        # For now, return None to try next restart/rank
                        return None

            # If loop finishes without success, try next restart
        return None

    # Try ranks from low to high
    for r in range(1, 55):
        if time.time() > deadline - 10:
            break
        result = try_rank_als(r)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
