import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Target Tensor T
    # T is defined such that C[i, k] = sum_j A[i, j] B[j, k]
    # We flatten A to vector a (size nm), B to vector b (size mp), C to vector c (size pn).
    # c[idx_c] = sum_{idx_a, idx_b} T[idx_c, idx_a, idx_b] a[idx_a] b[idx_b]
    # idx_a = i*m + j
    # idx_b = j*p + k
    # idx_c = k*n + i
    
    T_target = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                T_target[idx_c, idx_a, idx_b] = 1.0

    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # T[i_c, i_a, i_b] = sum_r W[i_c, r] U[i_a, r] V[i_b, r]
        # Using einsum: 'ir,ar,br -> iab'
        return np.einsum('ir,ar,br->iab', W, U, V, optimize=True)

    def error_func(params, rank):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U = params[0:size_U].reshape(n * m, rank)
        V = params[size_U:size_U + size_V].reshape(m * p, rank)
        W = params[size_U + size_V:].reshape(p * n, rank)
        
        T_calc = compute_tensor(U, V, W)
        diff = T_calc - T_target
        return np.sum(diff * diff)

    def verify_solution(U, V, W):
        T_calc = compute_tensor(U, V, W)
        return np.allclose(T_calc, T_target, atol=1e-9)

    def find_solution(rank, max_restarts=10, time_limit=100):
        start_time = time.time()
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random sparse integers
            # Higher density for smaller ranks?
            density = 0.3
            U_init = np.random.choice([-1, 0, 1], size=(n * m, rank), p=[0.15, 0.7, 0.15])
            V_init = np.random.choice([-1, 0, 1], size=(m * p, rank), p=[0.15, 0.7, 0.15])
            W_init = np.random.choice([-1, 0, 1], size=(p * n, rank), p=[0.15, 0.7, 0.15])
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                # Use L-BFGS-B
                res = opt.minimize(error_func, params_init, args=(rank,), method='L-BFGS-B', 
                                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    size_U = n * m * rank
                    size_V = m * p * rank
                    size_W = p * n * rank
                    
                    U_opt = res.x[0:size_U].reshape(n * m, rank)
                    V_opt = res.x[size_U:size_U + size_V].reshape(m * p, rank)
                    W_opt = res.x[size_U + size_V:].reshape(p * n, rank)
                    
                    # Round to nearest integer
                    U_int = np.round(U_opt).astype(int)
                    V_int = np.round(V_opt).astype(int)
                    W_int = np.round(W_opt).astype(int)
                    
                    if verify_solution(U_int, V_int, W_int):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                        
            except Exception as e:
                pass
                
        return None

    # Try ranks from 66 upwards (60 is a theoretical target, but 66 is safer for search)
    # We prioritize finding *any* rank < 72.
    for rank in [66, 67, 68, 69, 70, 71]:
        if time.time() - start_time > time_limit:
            break
            
        # Reduce restarts for higher ranks to save time
        max_restarts = 5 if rank < 68 else 3
        sol = find_solution(rank, max_restarts=max_restarts, time_limit=min(20, time_limit - (time.time() - start_time)))
        if sol:
            return {"U": sol[0], "V": sol[1], "W": sol[2]}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
