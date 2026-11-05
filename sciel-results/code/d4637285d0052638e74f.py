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
    # T[u, v, w] = 1 if u=i*M+j, v=j*P+k, w=k*N+i
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
        
        # Compute reconstructed tensor T_rec[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Compare with the target tensor T
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

    def make_initialization(r, seed):
        """Create diverse initializations for rank r."""
        np.random.seed(seed)
        
        # Strategy 1: Random structured initialization
        # For matrix multiplication, each term corresponds to a product of entries
        # Try random linear combinations that might capture structure
        
        U_init = np.random.randn(N, M, r) * 0.3
        V_init = np.random.randn(M, P, r) * 0.3
        W_init = np.random.randn(P, N, r) * 0.3
        
        # Add some structure: make certain entries more likely to be non-zero
        # based on the tensor pattern
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    # Each (i,j,k) triple should be represented
                    # Add a small bias to encourage coverage
                    if np.random.random() < 0.1:
                        c = np.random.randint(r)
                        U_init[i, j, c] += 1.0
                        V_init[j, k, c] += 1.0
                        W_init[k, i, c] += 1.0
        
        return U_init.flatten(), V_init.flatten(), W_init.flatten()
    
    def polish_solution(U, V, W, r, max_iter=200):
        """High-precision polish pass with tighter tolerances."""
        U_flat = U.flatten().copy()
        V_flat = V.flatten().copy()
        W_flat = W.flatten().copy()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                args=(r,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 30}
            )
            if res.fun < 1e-10:
                return res.x
        except Exception:
            pass
        
        return x0

    def try_rank(r, num_seeds=15, max_iter=300):
        """Try to find a decomposition of rank r."""
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Get initialization
            U_init, V_init, W_init = make_initialization(r, seed * 100 + r)
            x0 = np.concatenate([U_init, V_init, W_init])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
            except Exception:
                continue
            
            if res.fun < 1e-4:
                # Extract solution
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(NM, r)
                V = V_flat.reshape(MP, r)
                W = W_flat.reshape(PN, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if check_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Try polish pass if close
                if res.fun < 1e-2:
                    polished_x = polish_solution(U, V, W, r)
                    if polished_x is not None:
                        U_p = polished_x[:size_U].reshape(NM, r)
                        V_p = polished_x[size_U:size_U + size_V].reshape(MP, r)
                        W_p = polished_x[size_U + size_V:].reshape(PN, r)
                        
                        U_p_rounded = np.round(U_p * 2) / 2
                        V_p_rounded = np.round(V_p * 2) / 2
                        W_p_rounded = np.round(W_p * 2) / 2
                        
                        U_p_list = U_p_rounded.tolist()
                        V_p_list = V_p_rounded.tolist()
                        W_p_list = W_p_rounded.tolist()
                        
                        if check_exact(U_p_list, V_p_list, W_p_list):
                            return {"U": U_p_list, "V": V_p_list, "W": W_p_list}
        
        return None
    
    # Try ranks from lower to higher, with more seeds for lower ranks
    # Start from a reasonable lower bound and work up
    # For 3x4x6, try ranks 50-71
    for r in range(50, 72):
        if time.time() - start_time > DEADLINE:
            break
        
        # Use more seeds for lower ranks (harder to find)
        num_seeds = 20 if r < 60 else 15
        max_iter = 400 if r < 60 else 300
        
        result = try_rank(r, num_seeds=num_seeds, max_iter=max_iter)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If no better found, return schoolbook
    if best_rank >= N * M * P:
        return schoolbook()
    
    return best_result
