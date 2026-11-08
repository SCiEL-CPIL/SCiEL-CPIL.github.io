import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

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

def _find_3x3x2_rank_16(max_time=100.0):
    """
    Find a rank 16 decomposition for 3x3x2 matrix multiplication.
    n=3, m=3, p=2. U:(9, 16), V:(6, 16), W:(6, 16).
    """
    n, m, p = 3, 3, 2
    rank = 16
    start_time = time.time()
    
    # Objective function
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

    # Try multiple random seeds
    for seed in range(50):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if result.fun < 1e-10:
                Uf = result.x[:n*m*rank].reshape(n*m, rank)
                Vf = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                Wf = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Try to round
                    U_rounded = _to_half_int(Uf)
                    V_rounded = _to_half_int(Vf)
                    W_rounded = _to_half_int(Wf)
                    if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                        return U_rounded, V_rounded, W_rounded
                    return Uf, Vf, Wf
        except Exception:
            continue
    
    return None, None, None

def _construct_3x3x6_from_3x3x2(U2, V2, W2, rank2=16):
    """
    Construct 3x3x6 decomposition from three 3x3x2 decompositions.
    Total rank = 3 * rank2.
    """
    n, m, p = 3, 3, 6
    rank = 3 * rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for g in range(3):
        for r in range(rank2):
            col = g * rank2 + r
            U[:, col] = U2[:, r]
            for j in range(m):
                for local_k in range(2):
                    global_k = 2 * g + local_k
                    V[j * p + global_k, col] = V2[j * 2 + local_k, r]
            for local_k in range(2):
                for i in range(n):
                    global_k = 2 * g + local_k
                    W[global_k * n + i, col] = W2[local_k * n + i, r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Fallback schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Try to find rank 48 solution
    if time.time() - start_time < deadline - 10:
        U2, V2, W2 = _find_3x3x2_rank_16(max_time=100.0)
        if U2 is not None:
            U48, V48, W48 = _construct_3x3x6_from_3x3x2(U2, V2, W2, 16)
            err48 = _check_decomposition(U48, V48, W48, n, m, p)
            if err48 < 1e-8:
                best_rank = 48
                best_U, best_V, best_W = U48, V48, W48
        
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    return {"U": U_out, "V": V_out, "W": W_out}
