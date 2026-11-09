"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Build the matrix multiplication tensor
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T
    
    T = build_tensor()
    
    # Strassen 2x2 decomposition (rank 56)
    def strassen_decomposition():
        strassen_products = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        rank = 56
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for block in range(8):
            qc = block // 2
            ar = block % 2
            a_col0 = 2 * ar
            b_row0 = 2 * ar
            b_col0 = 2 * qc
            c_col0 = 2 * qc
            
            for u_coeff, v_coeff, w_coeff in strassen_products:
                for (li, lj), coeff in u_coeff.items():
                    gi = li
                    gj = a_col0 + lj
                    U[gi * m + gj, col] += coeff
                for (lj, lk), coeff in v_coeff.items():
                    gj = b_row0 + lj
                    gk = b_col0 + lk
                    V[gj * p + gk, col] += coeff
                for (lk, li), coeff in w_coeff.items():
                    gk = c_col0 + lk
                    gi = li
                    W[gk * n + gi, col] += coeff
                col += 1
        
        return U, V, W
    
    U_strassen, V_strassen, W_strassen = strassen_decomposition()
    
    def verify(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)
    
    assert verify(U_strassen, V_strassen, W_strassen), "Strassen verification failed"
    
    best_U, best_V, best_W = U_strassen, V_strassen, W_strassen
    best_rank = 56
    
    # Try to find a lower-rank decomposition using numerical optimization
    from scipy.optimize import minimize
    
    def try_rank(r, max_iter=20):
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        def objective(params):
            U_f = params[:n*m*r].reshape(n*m, r)
            V_f = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_f = params[n*m*r + m*p*r:].reshape(p*n, r)
            T_recon = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
            return np.sum((T_recon - T)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        
        U_opt = result.x[:n*m*r].reshape(n*m, r)
        V_opt = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result.x[n*m*r + m*p*r:].reshape(p*n, r)
        
        if verify(U_opt, V_opt, W_opt):
            return U_opt, V_opt, W_opt, True
        return None, None, None, False
    
    for r in [55, 54, 53, 52, 51, 50]:
        if time.time() > deadline - 40:
            break
        U_r, V_r, W_r, success = try_rank(r)
        if success:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r
            break
    
    # Convert to lists
    U_list = [[float(best_U[i,j]) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i,j]) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i,j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
