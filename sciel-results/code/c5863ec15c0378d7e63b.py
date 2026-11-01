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

    def try_rank_reduction(U0, V0, W0, T, target_rank, deadline, max_iters=200):
        """Try to reduce rank from current to target using gradient descent."""
        n, m, p = DIMENSIONS
        nr, mr, pr = n*m, m*p, p*n
        
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        # Project to target rank by keeping top singular values
        if U.shape[1] > target_rank:
            _, _, U_r = np.linalg.svd(U, full_matrices=False)
            U = U_r[:, :target_rank]
            _, _, V_r = np.linalg.svd(V, full_matrices=False)
            V = V_r[:, :target_rank]
            _, _, W_r = np.linalg.svd(W, full_matrices=False)
            W = W_r[:, :target_rank]
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            T_calc = np.einsum('at,bt,ct->abc', U, V, W)
            err = T_calc - T
            loss = np.sum(err**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                return best_U, best_V, best_W, True
            
            gU = np.einsum('abc,bt,ct->at', err, V, W)
            gV = np.einsum('abc,at,ct->bt', err, U, W)
            gW = np.einsum('abc,at,bt->ct', err, U, V)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.01 * (1.0 + 0.1 * np.sin(it * 0.1))
            
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
        
        # Check if we achieved target rank
        ok, final_err = verify_decomposition(best_U, best_V, best_W, T)
        if ok:
            return best_U, best_V, best_W, True
        return best_U, best_V, best_W, False

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
    
    # Strategy 1: Try direct rank reduction from 28 to lower ranks
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for restart in range(2):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(500 + target_rank * 7 + restart)
            
            # Start from Strassen and add small perturbation
            U0, V0, W0 = get_strassen_28()
            U_init = U0[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.05
            V_init = V0[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.05
            W_init = W0[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.05
            
            U_r, V_r, W_r, success = try_rank_reduction(
                U_init, V_init, W_init, T, target_rank, deadline, max_iters=150
            )
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            break
    
    # Strategy 2: If 27 didn't work, try a different approach - 
    # exploit the structure where A-matrices are shared across column pairs
    if best_rank == 28 and time.time() < deadline - 25:
        # The 2x2x2 Strassen has 7 terms. For 2x2x8, we have 4 independent 2x2x2 blocks.
        # But the A part is the same for all blocks! This means we might be able to
        # share some of the A factors.
        
        # Try a structured decomposition: use 7 A-factors but combine V and W
        # across the 4 column pairs more cleverly.
        
        for target_rank in [27, 26]:
            if time.time() > deadline - 15:
                break
            
            np.random.seed(900 + target_rank * 3)
            
            # Create a structured initialization
            U_struct = np.zeros((n*m, target_rank))
            V_struct = np.zeros((m*p, target_rank))
            W_struct = np.zeros((p*n, target_rank))
            
            strassen_products = [
                {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
                {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
                {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
                {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
                {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
                {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
                {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
            ]
            
            # For target_rank=27, we need to merge one pair
            # Use the first 26 terms as-is, and combine the last two
            if target_rank >= 26:
                num_full = min(7, (target_rank - 1) // 4)  # How many full blocks
                remaining = target_rank - num_full * 4
            
            # Simpler: just use gradient descent from a good start
            U0, V0, W0 = get_strassen_28()
            
            # Randomly select which columns to keep/merge
            for attempt in range(3):
                if time.time() > deadline - 12:
                    break
                
                np.random.seed(1000 + target_rank * 11 + attempt)
                
                U_init = U0[:, :target_rank] + np.random.randn(n*m, target_rank) * 0.1
                V_init = V0[:, :target_rank] + np.random.randn(m*p, target_rank) * 0.1
                W_init = W0[:, :target_rank] + np.random.randn(p*n, target_rank) * 0.1
                
                U_r, V_r, W_r, success = try_rank_reduction(
                    U_init, V_init, W_init, T, target_rank, deadline, max_iters=100
                )
                
                if success:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    break
            
            if best_rank < 28:
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
