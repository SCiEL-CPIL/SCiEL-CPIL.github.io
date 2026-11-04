import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor T[i,j,k] = 1 if consistent, else 0
    # For standard matrix multiplication, all indices are consistent.
    T = np.ones((n, m, p), dtype=np.float64)
    
    N, M, P = n, m, p
    
    def verify(U, V, W):
        """
        Check if U, V, W exactly decompose T.
        U: (N*M, r), V: (M*P, r), W: (P*N, r)
        """
        r = U.shape[1]
        U_reshaped = U.reshape(N, M, r)
        V_reshaped = V.reshape(M, P, r)
        W_reshaped = W.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check for exact match within tight tolerance
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T_rec - T
        loss = np.sum(err**2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def try_rank(r, seed=0):
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        # Initialization: Random small values
        x0 = np.random.randn(total_size) * 0.5
        
        # Use L-BFGS-B
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        except Exception:
            return None
            
        if time.time() > deadline:
            return None
            
        # Check if optimization converged to a low error
        if res.fun < 1e-6:
            U_opt = res.x[:size_U].reshape(N, M, r)
            V_opt = res.x[size_U:size_U+size_V].reshape(M, P, r)
            W_opt = res.x[size_U+size_V:].reshape(P, N, r)
            
            # Try snapping to integers
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            if verify(U_int, V_int, W_int):
                return U_int.flatten(axis=1), V_int.flatten(axis=1), W_int.flatten(axis=1)
                
            # Try snapping to half-integers
            U_half = np.round(U_opt * 2) / 2.0
            V_half = np.round(V_opt * 2) / 2.0
            W_half = np.round(W_opt * 2) / 2.0
            if verify(U_half, V_half, W_half):
                return U_half.flatten(axis=1), V_half.flatten(axis=1), W_half.flatten(axis=1)
            
            # Try snapping to 1/3 integers
            U_third = np.round(U_opt * 3) / 3.0
            V_third = np.round(V_opt * 3) / 3.0
            W_third = np.round(W_opt * 3) / 3.0
            if verify(U_third, V_third, W_third):
                return U_third.flatten(axis=1), V_third.flatten(axis=1), W_third.flatten(axis=1)

        return None
    
    # Fallback: Schoolbook
    U_fb = np.zeros((N*M, rank_schoolbook))
    V_fb = np.zeros((M*P, rank_schoolbook))
    W_fb = np.zeros((P*N, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_schoolbook
    
    # Search for lower rank
    # We prioritize ranks close to schoolbook.
    # We allocate more time to the first few ranks.
    ranks_to_try = [rank_schoolbook - 1, rank_schoolbook - 2, rank_schoolbook - 3]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple seeds for each rank
        # Use more seeds for the highest priority rank
        num_seeds = 10 if r == rank_schoolbook - 1 else 5
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
                # Found a valid one for this rank, try lower
                break
    
    U_out, V_out, W_out = best_result
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
