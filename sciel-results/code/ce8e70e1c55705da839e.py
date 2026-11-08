import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    deadline = _t0 + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 26 solution (from current program)
    def get_rank_26():
        rank = 26
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_idx, v_idx, w_idx, coeff):
            nonlocal col
            U[u_idx][col] += coeff
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

        # Strassen 2x2
        U[0][col] = 1; U[4][col] = 1; V[0][col] = 1; V[4][col] = 1; W[0][col] = 1; W[4][col] = 1; col += 1
        U[3][col] = 1; U[4][col] = 1; V[0][col] = 1; W[1][col] = 1; W[4][col] = -1; col += 1
        U[0][col] = 1; V[1][col] = 1; V[4][col] = -1; W[3][col] = 1; W[4][col] = 1; col += 1
        U[4][col] = 1; V[3][col] = 1; V[0][col] = -1; W[0][col] = 1; W[1][col] = 1; col += 1
        U[0][col] = 1; U[1][col] = 1; V[4][col] = 1; W[0][col] = -1; W[3][col] = 1; col += 1
        U[3][col] = 1; U[0][col] = -1; V[0][col] = 1; V[1][col] = 1; W[4][col] = 1; col += 1
        U[1][col] = 1; U[4][col] = -1; V[3][col] = 1; V[4][col] = 1; W[0][col] = 1; col += 1
        
        # Linear terms
        add_term(2, 6, 0, 1)
        add_term(2, 7, 3, 1)
        add_term(5, 6, 1, 1)
        add_term(5, 7, 4, 1)
        
        add_term(0, 2, 6, 1)
        add_term(1, 5, 6, 1)
        add_term(3, 2, 7, 1)
        add_term(4, 5, 7, 1)
        
        add_term(2, 8, 6, 1)
        add_term(5, 8, 7, 1)
        
        add_term(6, 0, 2, 1)
        add_term(7, 3, 2, 1)
        add_term(6, 1, 5, 1)
        add_term(7, 4, 5, 1)
        
        add_term(8, 6, 2, 1)
        add_term(8, 7, 5, 1)
        
        add_term(6, 2, 8, 1)
        add_term(7, 5, 8, 1)
        
        add_term(8, 8, 8, 1)
        
        return {"U": U, "V": V, "W": W}

    # Attempt to find Rank 23 decomposition
    try:
        rank = 23
        # Construct target tensor T
        T = np.zeros((9, 9, 9), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T[c, a, b] = 1.0
        
        T_flat = T.flatten()
        
        def residuals(x):
            U = x[0:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            T_calc = np.einsum('ar,br,cr->cab', U, V, W)
            return T_calc.flatten() - T_flat
        
        # Try multiple random initializations
        best_result = None
        best_cost = float('inf')
        
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 10:
                break
                
            np.random.seed(seed)
            # Scale initialization to be closer to expected magnitude
            x0 = np.random.randn(27 * rank) * 0.5
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
            except Exception:
                continue
        
        if best_result is not None and (best_result.success or best_result.cost < 1e-6):
            U_opt = best_result.x[0:9*rank].reshape(9, rank)
            V_opt = best_result.x[9*rank:18*rank].reshape(9, rank)
            W_opt = best_result.x[18*rank:27*rank].reshape(9, rank)
            
            # Round to nearest half-integer
            U_rounded = np.round(U_opt * 2) / 2.0
            V_rounded = np.round(V_opt * 2) / 2.0
            W_rounded = np.round(W_opt * 2) / 2.0
            
            # Verify exactness
            T_verify = np.einsum('ar,br,cr->cab', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_verify, T, atol=1e-9):
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    except Exception:
        pass

    # Fallback
    return get_rank_26()
# EVOLVE-BLOCK-END
