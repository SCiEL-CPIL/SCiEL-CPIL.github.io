"""Matrix multiplication tensor decomposition search for 3x4x6."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    global START_TIME
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    NM, MP, PN = N * M, M * P, P * N
    
    # Build target tensor T of shape (N*M, M*P, P*N)
    T = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook fallback
    def schoolbook():
        rank = N * M * P
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = N * M * P
    
    def check_exact(U_list, V_list, W_list):
        """Check if the decomposition exactly reconstructs T."""
        if not U_list:
            return False
        r = len(U_list[0])
        U = np.array(U_list)
        V = np.array(V_list)
        W = np.array(W_list)
        
        T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        T_3d = np.ones((N, M, P))
        err = T_rec - T_3d
        loss = np.sum(err ** 2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def try_rank(r, num_seeds=3, max_iter=100):
        """Try to find a decomposition of rank r."""
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            np.random.seed(seed * 1000 + r)
            x0 = np.random.randn(total_size) * 0.1
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 10}
                )
            except Exception:
                continue
            
            if res.fun < 1e-4:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(NM, r)
                V = V_flat.reshape(MP, r)
                W = W_flat.reshape(PN, r)
                
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if check_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Use tensor matricization SVD to find a good starting rank
    T_mat = T.reshape(NM, MP * PN)
    _, s_vals, _ = np.linalg.svd(T_mat, full_matrices=False)
    num_rank = np.sum(s_vals > 1e-10)
    
    # Also check other matricizations for tighter bounds
    T_mat2 = T.reshape(MP, NM * PN)
    _, s_vals2, _ = np.linalg.svd(T_mat2, full_matrices=False)
    num_rank2 = np.sum(s_vals2 > 1e-10)
    
    T_mat3 = T.reshape(PN, NM * MP)
    _, s_vals3, _ = np.linalg.svd(T_mat3, full_matrices=False)
    num_rank3 = np.sum(s_vals3 > 1e-10)
    
    lower_bound = max(num_rank, num_rank2, num_rank3)
    
    # Try ranks from lower bound (or slightly above) up to schoolbook
    min_rank_to_try = max(lower_bound, 15)
    max_rank_to_try = N * M * P - 1
    
    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_rank(r, num_seeds=3, max_iter=200)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    if best_rank >= N * M * P:
        return schoolbook()
    
    return best_result
