"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _find_3x3x2_rank7(max_time=30.0):
    """
    Find a rank-7 decomposition for 3x3x2 matrix multiplication.
    n=3, m=3, p=2. U:(9,7), V:(6,7), W:(6,7).
    """
    n, m, p = 3, 3, 2
    rank = 7
    start_time = time.time()
    
    for seed in range(50):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def obj(x):
            Uf = x[:n*m*rank].reshape(n*m, rank)
            Vf = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            Wf = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                        err += (val - 1.0)**2
            return err
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            
            if result.fun < 1e-12:
                Uf = result.x[:n*m*rank].reshape(n*m, rank)
                Vf = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                Wf = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    return Uf, Vf, Wf
        except Exception:
            continue
    
    return None, None, None

def _construct_3x3x6_from_3x3x2(U2, V2, W2):
    """
    Construct 3x3x6 decomposition from three 3x3x2 rank-7 decompositions.
    Total rank = 21.
    
    For 3x3x6: n=3, m=3, p=6.
    U:(9, 21), V:(18, 21), W:(18, 21).
    
    Split k=0..5 into three groups: {0,1}, {2,3}, {4,5}.
    For group g (g=0,1,2), local k indices are 0,1.
    Global k = 2*g + local_k.
    
    U[i*3+j, g*7+r] = U2[i*3+j, r]
    V[j*6 + (2*g + local_k), g*7+r] = V2[j*2 + local_k, r]
    W[(2*g + local_k)*3 + i, g*7+r] = W2[local_k*3 + i, r]
    """
    n, m, p = 3, 3, 6
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for g in range(3):
        for r in range(7):
            col = g * 7 + r
            # U: same for all groups
            U[:, col] = U2[:, r]
            # V: j*6 + (2*g + local_k)
            for j in range(m):
                for local_k in range(2):
                    global_k = 2 * g + local_k
                    V[j * p + global_k, col] = V2[j * 2 + local_k, r]
            # W: (2*g + local_k)*3 + i
            for local_k in range(2):
                for i in range(n):
                    global_k = 2 * g + local_k
                    W[global_k * n + i, col] = W2[local_k * n + i, r]
    
    return U, V, W

def _try_find_decomposition(n, m, p, target_rank, max_time=10.0):
    """
    Try to find a rank-target_rank decomposition using local optimization
    with structured initialization.
    """
    start_time = time.time()
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() - start_time > max_time:
            break
            
        U_init = np.random.randn(n * m, target_rank) * 0.5
        V_init = np.random.randn(m * p, target_rank) * 0.5
        W_init = np.random.randn(p * n, target_rank) * 0.5
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def obj(x):
            Uf = x[:n*m*target_rank].reshape(n*m, target_rank)
            Vf = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            Wf = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                        err += (val - 1.0)**2
            return err
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < 1e-10:
                U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
                
                err = _check_decomposition(U_found, V_found, W_found, n, m, p)
                if err < 1e-8:
                    return U_found, V_found, W_found
        except Exception:
            continue
            
    return None, None, None

def _high_precision_polish(U, V, W, n, m, p, max_iter=500):
    """
    High-precision polish step.
    """
    rank = U.shape[1]
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        Uf = x[:n*m*rank].reshape(n*m, rank)
        Vf = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        Wf = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.fun < 1e-12:
            U_polished = result.x[:n*m*rank].reshape(n * m, rank)
            V_polished = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
            W_polished = result.x[n*m*rank + m*p*rank :].reshape(p * n, rank)
            
            err = _check_decomposition(U_polished, V_polished, W_polished, n, m, p)
            if err < 1e-10:
                return U_polished, V_polished, W_polished
    except Exception:
        pass
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook is valid
    err = _check_decomposition(U, V, W, n, m, p)
    if err >= 1e-10:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Strategy 1: Try Winograd FFM - 3x3x6 as three 3x3x2 with rank 7 each = rank 21
    if time.time() - start_time < deadline - 10:
        U2, V2, W2 = _find_3x3x2_rank7(max_time=40.0)
        if U2 is not None:
            U21, V21, W21 = _construct_3x3x6_from_3x3x2(U2, V2, W2)
            err21 = _check_decomposition(U21, V21, W21, n, m, p)
            if err21 < 1e-8:
                # Polish
                U21, V21, W21 = _high_precision_polish(U21, V21, W21, n, m, p)
                err21 = _check_decomposition(U21, V21, W21, n, m, p)
                if err21 < 1e-8:
                    best_rank = 21
                    best_U, best_V, best_W = U21, V21, W21
                    # Try to round to half-integers
                    U_rounded = _to_half_int(best_U)
                    V_rounded = _to_half_int(best_V)
                    W_rounded = _to_half_int(best_W)
                    if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    U_out = [[float(x) for x in row] for row in best_U]
                    V_out = [[float(x) for x in row] for row in best_V]
                    W_out = [[float(x) for x in row] for row in best_W]
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # Strategy 2: Try direct optimization for ranks below 54
    for target_rank in range(50, 20, -1):
        if time.time() - start_time > deadline - 5:
            break
        
        max_time_per_rank = 8.0
        U_r, V_r, W_r = _try_find_decomposition(n, m, p, target_rank, max_time=max_time_per_rank)
        
        if U_r is not None:
            U_r, V_r, W_r = _high_precision_polish(U_r, V_r, W_r, n, m, p)
            
            U_rounded = _to_half_int(U_r)
            V_rounded = _to_half_int(V_r)
            W_rounded = _to_half_int(W_r)
            
            if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            else:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
            
            break
    
    # Final result
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
