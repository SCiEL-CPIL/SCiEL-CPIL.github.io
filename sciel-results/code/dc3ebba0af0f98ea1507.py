"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import scipy.optimize
import time

# EVOLVE-BLOCK-START
def _create_target_tensor(n, m, p):
    """Create the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _verify_decomposition(U, V, W, T_target, atol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    # U, V, W are numpy arrays
    # Res[I, J, K] = Sum_c U[I, c] V[J, c] W[K, c]
    Res = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(Res, T_target, atol=atol)

def _snap_to_halfint(x):
    """Snap values to the nearest half-integer."""
    return np.round(x * 2) / 2

def _search_rank(R, T_target, n, m, p, deadline):
    """Try to find a decomposition of rank R."""
    nm, mp, pn = n * m, m * p, p * n
    
    def residual(vec):
        U_flat = vec[:nm * R]
        V_flat = vec[nm * R : nm * R + mp * R]
        W_flat = vec[nm * R + mp * R :]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        Res = np.einsum('ic,jc,kc->ijk', U, V, W)
        return (Res - T_target).flatten()
    
    # Try a few random restarts
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(attempt * 100 + R)
        U_init = np.random.randn(nm, R) * 0.5
        V_init = np.random.randn(mp, R) * 0.5
        W_init = np.random.randn(pn, R) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.least_squares(residual, x0, method='trf', 
                                               ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
            
            if res.success or res.cost < 1e-8:
                U_opt = res.x[:nm * R].reshape(nm, R)
                V_opt = res.x[nm * R : nm * R + mp * R].reshape(mp, R)
                W_opt = res.x[nm * R + mp * R :].reshape(pn, R)
                
                # Snap to half-integers
                U_snap = _snap_to_halfint(U_opt)
                V_snap = _snap_to_halfint(V_opt)
                W_snap = _snap_to_halfint(W_opt)
                
                # Verify snapped solution
                if _verify_decomposition(U_snap, V_snap, W_snap, T_target, atol=1e-5):
                    return U_snap, V_snap, W_snap
                    
        except Exception:
            pass
            
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    T_target = _create_target_tensor(n, m, p)
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks lower than schoolbook
    # Known lower bound for 3x5x5 is likely around 60-70.
    # We'll try a range.
    ranks_to_try = list(range(74, 49, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        result = _search_rank(R, T_target, n, m, p, deadline)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = R
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            break  # Found the lowest rank in this sequence, stop
            
    # Convert to list of lists as required
    final_rank = best_U.shape[1]
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
