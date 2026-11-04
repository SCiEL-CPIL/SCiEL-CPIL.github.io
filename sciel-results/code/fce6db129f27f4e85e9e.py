import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook solution
    rank_schoolbook = n * m * p  # 30
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(total - 1) > tol:
                        return False
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    total = np.sum(U[a] * V[b] * W[c])
                    i_a, j_a = divmod(a, m)
                    j_b, k_b = divmod(b, p)
                    k_c, i_c = divmod(c, n)
                    expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                    if abs(total - expected) > tol:
                        return False
        return True
    
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Try lower ranks
    for R in range(rank_schoolbook - 1, 19, -1):
        if time.time() > deadline - 20:
            break
        
        # Initialize with a random perturbation of the schoolbook solution
        # Take the first R columns of the schoolbook solution and perturb
        U_init = U_sb[:, :R].copy()
        V_init = V_sb[:, :R].copy()
        W_init = W_sb[:, :R].copy()
        
        # Add small random perturbation
        U_init += 0.1 * np.random.randn(n*m, R)
        V_init += 0.1 * np.random.randn(m*p, R)
        W_init += 0.1 * np.random.randn(p*n, R)
        
        # Flatten for optimization
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            U = x[:n*m*R].reshape(n*m, R)
            V = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W = x[n*m*R+m*p*R:].reshape(p*n, R)
            return U, V, W
        
        def residuals(x):
            U, V, W = unflatten(x)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res.append(np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]) - 1)
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        i_a, j_a = divmod(a, m)
                        j_b, k_b = divmod(b, p)
                        k_c, i_c = divmod(c, n)
                        expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                        res.append(np.sum(U[a] * V[b] * W[c]) - expected)
            return np.array(res)
        
        x0 = flatten(U_init, V_init, W_init)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            U_opt, V_opt, W_opt = unflatten(result.x)
            
            # Round to half-integers
            U_rounded = round_to_half(U_opt)
            V_rounded = round_to_half(V_opt)
            W_rounded = round_to_half(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = R
                break  # Found a valid solution, stop searching
        except Exception as e:
            pass
    
    return best_result
