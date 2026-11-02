import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    return np.allclose(T_computed, T_target, atol=1e-9)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110
    
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    best = {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
    best_rank = rank_trivial
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    T_target_flat = T_target.flatten()
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    total_size = size_U + size_V + size_W
    
    def residual(x):
        R = len(x) // total_size
        U = x[:size_U*R].reshape(size_U, R)
        V = x[size_U*R : size_U*R + size_V*R].reshape(size_V, R)
        W = x[size_U*R + size_V*R :].reshape(size_W, R)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return T.flatten() - T_target_flat
        
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline:
            break
            
        for trial in range(10):
            if time.time() > deadline:
                break
                
            x0 = np.random.randint(-1, 2, total_size * target_rank)
            
            try:
                res = least_squares(residual, x0, method='lm', max_nfev=2000)
                
                if res.cost < 1e-6:
                    R = target_rank
                    U = res.x[:size_U*R].reshape(size_U, R)
                    V = res.x[size_U*R : size_U*R + size_V*R].reshape(size_V, R)
                    W = res.x[size_U*R + size_V*R :].reshape(size_W, R)
                    
                    # Try rounding to integers
                    candidates = [
                        (np.round(U), np.round(V), np.round(W)),
                        (np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2)
                    ]
                    
                    for U_c, V_c, W_c in candidates:
                        if verify_decomposition(U_c, V_c, W_c, n, m, p):
                            best = {'U': U_c.tolist(), 'V': V_c.tolist(), 'W': W_c.tolist()}
                            best_rank = target_rank
                            print(f"Found rank {best_rank}")
                            break
            except Exception:
                pass
                
    return best
