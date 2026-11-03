import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Target tensor indices
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try to find a low rank decomposition
    # Start with a lower rank bound, e.g., 32
    for rank in [32, 40, 50, 60, 70]:
        np.random.seed(42)
        # Initialize with complex values
        U = (np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)) * 0.1
        V = (np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)) * 0.1
        W = (np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)) * 0.1
        
        # Flatten parameters for optimization
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_ = params[:nm*rank].reshape(nm, rank)
            V_ = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_ = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Compute tensor
            T = np.einsum('ar,br,cr->abc', U_, V_, W_)
            
            # Compute loss (Frobenius norm of residual)
            # Only non-zero entries of target are 1, others are 0
            # We can compute the full tensor and compare
            residual = T - T_target
            return np.sum(np.abs(residual)**2)
        
        # Use L-BFGS-B or BFGS
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
        
        if res.success or res.fun < 1e-10:
            # Extract optimized parameters
            U_opt = res.x[:nm*rank].reshape(nm, rank)
            V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Snap to half-integers
            # Since the solution should be real if we started with symmetric complex or if the manifold is real
            # We take the real part if imaginary part is small
            if np.max(np.abs(U_opt.imag)) < 1e-6 and np.max(np.abs(V_opt.imag)) < 1e-6 and np.max(np.abs(W_opt.imag)) < 1e-6:
                U_real = U_opt.real
                V_real = V_opt.real
                W_real = W_opt.real
                
                U_snap = np.round(U_real * 2) / 2
                V_snap = np.round(V_real * 2) / 2
                W_snap = np.round(W_real * 2) / 2
                
                # Verify
                T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                error = np.sum((T_snap - T_target)**2)
                
                if error < 1e-10:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            else:
                # If complex, try to find a real solution by re-optimizing with real init
                # Or just fail this rank
                pass
        
    # Fallback to schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
