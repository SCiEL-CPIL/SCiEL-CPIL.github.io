import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    n, m, p = 2, 3, 6

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        # Reshape to 3D for einsum
        U3 = U.reshape(n_s, m_s, r)
        V3 = V.reshape(m_s, p_s, r)
        W3 = W.reshape(p_s, n_s, r)
        # Compute T
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Check if T is all ones
        return np.allclose(T, np.ones((n_s, m_s, p_s)), atol=1e-9)

    def compute_residuals_flat(x, rank):
        """
        x is a flat vector of size (n*m + m*p + p*n) * rank
        """
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U = x[:size_U].reshape(n, m, rank)
        V = x[size_U:size_U + size_V].reshape(m, p, rank)
        W = x[size_U + size_V:].reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        residuals = T.flatten() - 1.0
        return residuals

    def try_rank(rank, max_seeds=20, max_iter=1000):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W

        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                return None

            np.random.seed(seed * 1000 + rank + 42)
            x0 = np.random.randn(total_size)
            
            # Scale initial vectors to avoid large/small values
            # Normalize each "column" (index r)
            U0 = x0[:size_U].reshape(n, m, rank)
            V0 = x0[size_U:size_U + size_V].reshape(m, p, rank)
            W0 = x0[size_U + size_V:].reshape(p, n, rank)
            
            for c in range(rank):
                nU = np.linalg.norm(U0[:, :, c])
                nV = np.linalg.norm(V0[:, :, c])
                nW = np.linalg.norm(W0[:, :, c])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    scale = (nU * nV * nW) ** (1.0/3.0)
                    U0[:, :, c] /= scale
                    V0[:, :, c] /= scale
                    W0[:, :, c] /= scale
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            def fun(x):
                return compute_residuals_flat(x, rank)

            try:
                # Use Levenberg-Marquardt
                res = opt.least_squares(fun, x0, method='lm', max_nfev=500)
                err = np.sum(res.fun**2)
                
                if err < 1e-8:
                    U_opt = res.x[:size_U].reshape(n, m, rank)
                    V_opt = res.x[size_U:size_U + size_V].reshape(m, p, rank)
                    W_opt = res.x[size_U + size_V:].reshape(p, n, rank)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Flatten back to 2D for return format
                    U_2d = U_snap.reshape(n * m, rank)
                    V_2d = V_snap.reshape(m * p, rank)
                    W_2d = W_snap.reshape(p * n, rank)
                    
                    if verify_full(U_2d, V_2d, W_2d, n, m, p):
                        return U_2d, V_2d, W_2d
                        
                    # Try integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    U_2d_int = U_int.reshape(n * m, rank)
                    V_2d_int = V_int.reshape(m * p, rank)
                    W_2d_int = W_int.reshape(p * n, rank)
                    if verify_full(U_2d_int, V_2d_int, W_2d_int, n, m, p):
                        return U_2d_int, V_2d_int, W_2d_int
                        
            except Exception:
                pass

        return None

    # Try ranks from theoretical minimum upwards
    # Theoretical lower bound for 2x3x6 is 16.
    # We try 16 first. If it fails, we try higher.
    for rank in range(16, 37):
        if time.time() > deadline - 10:
            break
            
        result = try_rank(rank)
        if result:
            U_final, V_final, W_final = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            # If we found the theoretical minimum, we can stop
            if rank == 16:
                break

    return best_result
# EVOLVE-BLOCK-END
