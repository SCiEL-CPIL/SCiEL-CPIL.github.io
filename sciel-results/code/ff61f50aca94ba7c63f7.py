import numpy as np
import time
import random

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_term(A_coeffs, B_coeffs, C_coeffs, k_start):
        nonlocal col_idx
        r = col_idx
        col_idx += 1
        
        for i, j, c in A_coeffs:
            U[i * m + j][r] = c
        for j, k_off, c in B_coeffs:
            V[j * p + (k_start + k_off)][r] = c
        for i, k_off, c in C_coeffs:
            W[(k_start + k_off) * n + i][r] = c

    S1_A = [(0,0,1), (1,1,1)]
    S1_B = [(0,0,1), (1,1,1)]
    S1_C = [(0,0,1), (1,1,1)]
    
    S2_A = [(1,0,1), (1,1,1)]
    S2_B = [(0,0,1)]
    S2_C = [(1,0,1), (1,1,-1)]
    
    S3_A = [(0,0,1)]
    S3_B = [(0,1,1), (1,1,-1)]
    S3_C = [(0,1,1), (1,1,1)]
    
    S4_A = [(1,1,1)]
    S4_B = [(1,0,1), (0,0,-1)]
    S4_C = [(0,0,1), (1,0,1)]
    
    S5_A = [(0,0,1), (0,1,1)]
    S5_B = [(1,1,1)]
    S5_C = [(0,0,-1), (0,1,1)]
    
    S6_A = [(1,0,1), (0,0,-1)]
    S6_B = [(0,0,1), (0,1,1)]
    S6_C = [(1,1,1)]
    
    S7_A = [(0,1,1), (1,1,-1)]
    S7_B = [(1,0,1), (1,1,1)]
    S7_C = [(0,0,1)]

    strassen_terms = [
        (S1_A, S1_B, S1_C),
        (S2_A, S2_B, S2_C),
        (S3_A, S3_B, S3_C),
        (S4_A, S4_B, S4_C),
        (S5_A, S5_B, S5_C),
        (S6_A, S6_B, S6_C),
        (S7_A, S7_B, S7_C)
    ]
    
    for b in range(4):
        k_start = b * 2
        for A_c, B_c, C_c in strassen_terms:
            add_strassen_term(A_c, B_c, C_c, k_start)
            
    k_start = 8
    add_strassen_term([(0,0,1)], [(0,0,1)], [(0,0,1)], k_start)
    add_strassen_term([(0,1,1)], [(1,0,1)], [(0,0,1)], k_start)
    add_strassen_term([(1,0,1)], [(0,0,1)], [(1,0,1)], k_start)
    add_strassen_term([(1,1,1)], [(1,0,1)], [(1,0,1)], k_start)

    return {"U": U, "V": V, "W": W}

def _try_rank30_search():
    # Attempt to find rank-10 decomposition for M_{2,2,3}
    # If found, construct rank-30 for M_{2,2,9}
    n, m, p_block = 2, 2, 3
    rank_block = 10
    
    # Tensor T for M_{2,2,3}
    # U shape (4, 10), V shape (6, 10), W shape (6, 10)
    # Target: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    # Indices: U: i*m+j (0..3), V: j*p+k (0..5), W: k*n+i (0..5)
    
    # We define the target tensor values
    # T_alpha_beta_gamma = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # else 0.
    
    # We will use gradient descent on the residual
    # Residual = sum (sum_r U_alpha_r V_beta_r W_gamma_r - T_alpha_beta_gamma)^2
    
    # Flatten parameters
    # U: 4*10, V: 6*10, W: 6*10
    # Total params: 160
    
    def compute_residual(params):
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        # Compute product for all alpha, beta, gamma
        # This is a 4x6x6 tensor
        # P[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
        # We can do this efficiently
        # P = np.einsum('ar,br,cr->abc', U, V, W)
        P = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target T
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    alpha = i * 2 + j
                    beta = j * 3 + k
                    gamma = k * 2 + i
                    T[alpha, beta, gamma] = 1.0
        
        diff = P - T
        return np.sum(diff**2)

    def get_solution(params):
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        return U, V, W

    # Try a few random initializations
    best_params = None
    best_res = float('inf')
    
    start_time = time.time()
    time_limit = 10.0 # 10 seconds for search
    
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
            
        params = np.random.randn(160) * 0.5
        
        # Simple gradient descent or use scipy
        # Let's use scipy.optimize.minimize with L-BFGS-B
        from scipy.optimize import minimize
        
        res = minimize(compute_residual, params, method='L-BFGS-B', options={'maxiter': 500})
        
        if res.fun < best_res:
            best_res = res.fun
            best_params = res.x
            
        if best_res < 1e-10:
            break
            
    if best_res < 1e-10 and best_params is not None:
        U, V, W = get_solution(best_params)
        
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded solution is exact
        P_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    alpha = i * 2 + j
                    beta = j * 3 + k
                    gamma = k * 2 + i
                    T[alpha, beta, gamma] = 1.0
        
        if np.allclose(P_rounded, T, atol=1e-9):
            # Construct full rank 30 solution
            n, m, p = DIMENSIONS
            rank = 30
            U_full = [[0] * rank for _ in range(n * m)]
            V_full = [[0] * rank for _ in range(m * p)]
            W_full = [[0] * rank for _ in range(p * n)]
            
            for b in range(3):
                k_start = b * 3
                # Copy U, V, W for this block
                # U is same for all blocks in this strategy?
                # No, U indices are i*m+j.
                # In the block decomposition, U is shared?
                # If we use the same U for all blocks, we need to ensure the rank-1 terms don't interfere.
                # But V and W are block-local (zeros outside).
                # So we can just copy the same U columns for each block?
                # Wait, if we use the same U, the rank is not additive.
                # But the format requires separate columns for each rank-1 term.
                # So we must duplicate U for each block.
                
                for r in range(10):
                    col = b * 10 + r
                    
                    for i in range(2):
                        for j in range(2):
                            U_full[i * m + j][col] = U_rounded[i * m + j, r]
                            
                    for j in range(2):
                        for k in range(3):
                            V_full[j * p + (k_start + k)][col] = V_rounded[j * 3 + k, r]
                            
                    for k in range(3):
                        for i in range(2):
                            W_full[(k_start + k) * n + i][col] = W_rounded[k * 2 + i, r]
                            
            return {"U": U_full, "V": V_full, "W": W_full}
            
    return None

def construct():
    # Try to find rank 30
    sol = _try_rank30_search()
    if sol is not None:
        return sol
        
    # Fallback to rank 32
    return _build_rank32()
# EVOLVE-BLOCK-END
