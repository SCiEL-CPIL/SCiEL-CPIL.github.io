"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    # Build target tensor
    T_3d = np.ones((n, m, p))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reconstructs T."""
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_rec, T_3d, atol=tol)
    
    def try_rank_structured(r, max_iter=500, num_seeds=5):
        """Try to find a decomposition of rank r using structured initialization."""
        size_U = NM * r
        size_V = MP * r
        size_W = PN * r
        
        best_loss = np.inf
        best_x = None
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                return None
            
            np.random.seed(seed * 7919 + r * 31)
            
            # Structured initialization: try to use low-rank structure
            # Initialize with small random values
            U_init = np.random.randn(n, m, r) * 0.3
            V_init = np.random.randn(m, p, r) * 0.3
            W_init = np.random.randn(p, n, r) * 0.3
            
            # Try different initialization strategies
            if seed == 0:
                # Symmetric-like initialization
                U_init = np.ones((n, m, r)) / (r ** 0.5)
                V_init = np.ones((m, p, r)) / (r ** 0.5)
                W_init = np.ones((p, n, r)) / (r ** 0.5)
            elif seed == 1:
                # Random with scaling
                U_init = np.random.randn(n, m, r) * 0.5
                V_init = np.random.randn(m, p, r) * 0.5
                W_init = np.random.randn(p, n, r) * 0.5
            elif seed == 2:
                # Block structure
                for c in range(r):
                    U_init[:, :, c] = np.random.randn(n, m) * 0.2
                    V_init[:, :, c] = np.random.randn(m, p) * 0.2
                    W_init[:, :, c] = np.random.randn(p, n) * 0.2
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_flat = x[:size_U]
                V_flat = x[size_U:size_U + size_V]
                W_flat = x[size_U + size_V:]
                
                U = U_flat.reshape(n, m, r)
                V = V_flat.reshape(m, p, r)
                W = W_flat.reshape(p, n, r)
                
                T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = T_rec - T_3d
                loss = np.sum(err ** 2)
                return loss
            
            def gradient(x):
                U_flat = x[:size_U]
                V_flat = x[size_U:size_U + size_V]
                W_flat = x[size_U + size_V:]
                
                U = U_flat.reshape(n, m, r)
                V = V_flat.reshape(m, p, r)
                W = W_flat.reshape(p, n, r)
                
                T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = T_rec - T_3d
                
                grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
                grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
                grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                res = minimize(
                    objective,
                    x0,
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = res.x.copy()
                    
                    if res.fun < 1e-10:
                        break
            except Exception:
                continue
        
        if best_x is None or best_loss > 1e-8:
            return None
        
        U_flat = best_x[:size_U]
        V_flat = best_x[size_U:size_U + size_V]
        W_flat = best_x[size_U + size_V:]
        
        U = U_flat.reshape(NM, r)
        V = V_flat.reshape(MP, r)
        W = W_flat.reshape(PN, r)
        
        # Try rounding to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if check_exact(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        if check_exact(U_int, V_int, W_int):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        # Check without rounding (if values are already close to half-integers)
        if check_exact(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from just below schoolbook downward
    # Lower bound: max(n*m, m*p, p*n) = 24, but tensor rank is typically higher
    # For 3x4x6, try from 71 down
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try a range of ranks
    for r in range(71, 25, -1):
        if time.time() > deadline:
            break
        
        result = try_rank_structured(r, max_iter=300, num_seeds=3)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    if best_rank >= n * m * p:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
