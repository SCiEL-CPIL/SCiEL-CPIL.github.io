import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    def get_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        strassen_products = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
        ]
        
        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    k_global = k1 + k_sub
                    V[j * p + k_global, t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    k_global = k1 + k_sub
                    W[k_global * n + i, t] = coeff
        return U, V, W

    def verify_decomposition(U, V, W, T, tol=1e-8):
        T_calc = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.max(np.abs(T_calc - T))
        return err < tol, err

    def alternating_minimization(U, V, W, T, deadline, max_iters=500):
        """Alternating minimization with exact rank constraints."""
        nr, mr, pr, rank = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: fix V, W
            # T = U @ (V ⊙ W) where ⊙ is Khatri-Rao product
            VW = np.einsum('bt,ct->bc', V, W)
            # Reshape T to (nr, mr*pr)
            T_2d = T.reshape(nr, mr * pr)
            # Solve U @ VW = T_2d
            # U = T_2d @ VW^T @ (VW @ VW^T)^{-1}
            try:
                VW_pinv = np.linalg.pinv(VW)
                U_new = T_2d @ VW_pinv
            except np.linalg.LinAlgError:
                U_new = U
            
            # Update V: fix U, W
            UW = np.einsum('at,ct->ac', U, W)
            T_2d_v = T.transpose(1, 0, 2).reshape(mr, nr * pr)
            try:
                UW_pinv = np.linalg.pinv(UW)
                V_new = T_2d_v @ UW_pinv
            except np.linalg.LinAlgError:
                V_new = V
            
            # Update W: fix U, V
            UV = np.einsum('at,bt->ab', U, V)
            T_2d_w = T.transpose(2, 0, 1).reshape(pr, nr * mr)
            try:
                UV_pinv = np.linalg.pinv(UV)
                W_new = T_2d_w @ UV_pinv
            except np.linalg.LinAlgError:
                W_new = W
            
            U, V, W = U_new, V_new, W_new
            
            T_calc = np.einsum('at,bt,ct->abc', U, V, W)
            err = T_calc - T
            loss = np.sum(err**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
        
        ok, final_err = verify_decomposition(best_U, best_V, best_W, T)
        return best_U, best_V, best_W, ok

    def try_random_init(T, target_rank, deadline, max_restarts=5):
        """Try multiple random initializations with alternating minimization."""
        nr, mr, pr = n*m, m*p, p*n
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(2000 + target_rank * 13 + restart)
            
            # Scale: entries should be O(1)
            U_init = np.random.randn(nr, target_rank) * 0.5
            V_init = np.random.randn(mr, target_rank) * 0.5
            W_init = np.random.randn(pr, target_rank) * 0.5
            
            U_r, V_r, W_r, success = alternating_minimization(
                U_init, V_init, W_init, T, deadline, max_iters=300
            )
            
            if success:
                return U_r, V_r, W_r, True
        
        return None, None, None, False

    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = 145.0
    
    best_rank = 28
    best_U, best_V, best_W = get_strassen_28()
    
    # Try to find lower rank decompositions
    for target_rank in [27, 26, 25, 24, 23, 22]:
        if time.time() > deadline - 20:
            break
        
        # First try from Strassen with perturbation
        U0, V0, W0 = get_strassen_28()
        
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(3000 + target_rank * 17 + attempt)
            
            U_init = U0[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.1
            V_init = V0[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.1
            W_init = W0[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.1
            
            U_r, V_r, W_r, success = alternating_minimization(
                U_init, V_init, W_init, T, deadline, max_iters=400
            )
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank < 28:
            break
        
        # If that didn't work, try random initialization
        if time.time() < deadline - 15:
            U_r, V_r, W_r, success = try_random_init(T, target_rank, deadline, max_restarts=3)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # Final polish: if we have a solution, try to round to half-integers
    if best_rank < 28:
        U_final = np.round(best_U * 2) / 2
        V_final = np.round(best_V * 2) / 2
        W_final = np.round(best_W * 2) / 2
        
        ok, _ = verify_decomposition(U_final, V_final, W_final, T)
        if ok:
            best_U, best_V, best_W = U_final, V_final, W_final
        else:
            # Try rounding to integers
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            ok_int, _ = verify_decomposition(U_int, V_int, W_int, T)
            if ok_int:
                best_U, best_V, best_W = U_int, V_int, W_int
    
    U, V, W = best_U, best_V, best_W
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
