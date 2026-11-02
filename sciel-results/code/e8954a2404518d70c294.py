import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T[a,b,c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if indices are consistent, else 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        S = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(S - T)) < tol
    
    def snap_to_half_integers(U, V, W, scale=2):
        """Snap values to nearest half-integer"""
        U_s = np.round(U * scale) / scale
        V_s = np.round(V * scale) / scale
        W_s = np.round(W * scale) / scale
        return U_s, V_s, W_s
    
    # Strategy 1: Try to find a low-rank solution using gradient-based optimization
    # with multiple random restarts and annealing
    
    def residual(params, R):
        """Compute residual for given parameters and rank R"""
        size_U = nm * R
        size_V = mp * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((nm, R))
        V = V_flat.reshape((mp, R))
        W = W_flat.reshape((pn, R))
        
        S = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(R):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return (S - T).flatten()
    
    # Try constructive Winograd-like approach first
    # For the middle dimension of size 4, use 4-point evaluation
    
    # Let's try a direct constructive method:
    # Use the fact that we can factor the computation using polynomial evaluation
    
    # For 3x4x7, try rank 50-70 with optimization
    
    best_rank = nm * mp  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((nm, nm*mp))
    V_sb = np.zeros((mp, nm*mp))
    W_sb = np.zeros((pn, nm*mp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = nm * mp
    
    # Try to find better solutions
    from scipy.optimize import least_squares
    
    # Try ranks from 60 down to 40
    for R in range(65, 35, -1):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(8):
            if time.time() > deadline:
                break
            
            size_U = nm * R
            size_V = mp * R
            size_W = pn * R
            total_size = size_U + size_V + size_W
            
            if restart == 0:
                # Warm start: take first R columns of schoolbook
                U_init = U_sb[:, :R].copy()
                V_init = V_sb[:, :R].copy()
                W_init = W_sb[:, :R].copy()
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            elif restart == 1:
                # Symmetric initialization
                U_init = np.random.randn(nm, R) * 0.3
                V_init = np.random.randn(mp, R) * 0.3
                W_init = np.random.randn(pn, R) * 0.3
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Different random scales
                scale = 0.1 + 0.1 * restart
                U_init = np.random.randn(nm, R) * scale
                V_init = np.random.randn(mp, R) * scale
                W_init = np.random.randn(pn, R) * scale
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residual, x0, args=(R,), method='lm', max_nfev=500)
                
                if res.cost < 1e-12:
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U + size_V]
                    W_flat = res.x[size_U + size_V:]
                    
                    U_sol = U_flat.reshape((nm, R))
                    V_sol = V_flat.reshape((mp, R))
                    W_sol = W_flat.reshape((pn, R))
                    
                    # Try snapping to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_sol, V_sol, W_sol)
                    
                    if verify(U_snap, V_snap, W_snap):
                        best_rank = R
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found = True
                        break
                    else:
                        # Check if unsnapped is valid
                        if verify(U_sol, V_sol, W_sol):
                            # Try to find a rational representation
                            # For now, accept if very close
                            best_rank = R
                            best_U, best_V, best_W = U_sol, V_sol, W_sol
                            found = True
                            break
            except Exception:
                continue
        
        if found:
            break
    
    # If no good solution found, return schoolbook
    if best_rank == nm * mp:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    
    # Convert to lists, ensuring integer/half-integer entries
    def to_list(M):
        L = []
        for row in M:
            L.append([float(round(x * 2) / 2) for x in row])
        return L
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
