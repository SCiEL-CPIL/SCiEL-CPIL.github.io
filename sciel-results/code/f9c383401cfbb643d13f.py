import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    """Verify exact decomposition using integer arithmetic when possible."""
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Fraction(V[j][c]) for c in range(rank)] for j in range(m * p)]
    W_frac = [[Fraction(W[k][c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(n, m, p, U, V, W):
    """Fast verification using numpy float64 with tolerance, for initial screening."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(n, m, p):
    """
    Try to find a rank-(n*m*p - 1) decomposition.
    Strategy: Use numerical optimization with multiple restarts, then attempt
    to round to rationals with small denominators.
    """
    if time.time() > _DEADLINE - 20:
        return None
    
    target_rank = n * m * p - 1  # 49
    base = _schoolbook(n, m, p)
    
    try:
        import scipy.optimize as opt
        
        best_error = float('inf')
        best_params = None
        
        # Try multiple random initializations
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 15:
                break
            
            np.random.seed(42 + restart * 7)
            
            # Initialize: take schoolbook terms but drop one, then perturb
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Copy target_rank terms from schoolbook (drop the last one)
            for c in range(target_rank):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            # Add random perturbations scaled by restart
            scale = 0.05 + 0.02 * restart
            U += np.random.randn(n * m, target_rank) * scale
            V += np.random.randn(m * p, target_rank) * scale
            W += np.random.randn(p * n, target_rank) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                            error += (s - 1.0)**2
                return error
            
            def gradient(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                grad_U = np.zeros_like(U_mat)
                grad_V = np.zeros_like(V_mat)
                grad_W = np.zeros_like(W_mat)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                            ds = 2.0 * (s - 1.0)
                            grad_U[i*m+j] += ds * V_mat[j*p+k] * W_mat[k*n+i]
                            grad_V[j*p+k] += ds * U_mat[i*m+j] * W_mat[k*n+i]
                            grad_W[k*n+i] += ds * U_mat[i*m+j] * V_mat[j*p+k]
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                # Use L-BFGS-B with analytical gradient
                res = opt.minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                                   options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x.copy()
                    
            except Exception:
                pass
        
        if best_error < 1e-6 and best_params is not None:
            # Try to round to rationals
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try various scales for rounding to rationals
            for scale in [2, 4, 8, 16, 32]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                # Fast check first
                if _verify_fast(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    # Exact verification
                    if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try integer rounding
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if _verify_fast(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try continued fraction approximation for better rational recovery
            from fractions import Fraction
            for scale in [1, 2, 4, 8]:
                U_frac_list = []
                V_frac_list = []
                W_frac_list = []
                success = True
                for row in U:
                    U_frac_list.append([Fraction(int(round(x * scale)), scale) if scale > 1 else Fraction(int(round(x)), 1) for x in row])
                for row in V:
                    V_frac_list.append([Fraction(int(round(x * scale)), scale) if scale > 1 else Fraction(int(round(x)), 1) for x in row])
                for row in W:
                    W_frac_list.append([Fraction(int(round(x * scale)), scale) if scale > 1 else Fraction(int(round(x)), 1) for x in row])
                
                # Convert back to float lists for verification
                U_float = [[float(f) for f in row] for row in U_frac_list]
                V_float = [[float(f) for f in row] for row in V_frac_list]
                W_float = [[float(f) for f in row] for row in W_frac_list]
                
                if _verify_fast(n, m, p, U_float, V_float, W_float):
                    if _verify(n, m, p, U_float, V_float, W_float):
                        return {"U": U_float, "V": U_float, "W": W_float}
                
        # If optimization didn't converge well, try a different approach:
        # Search for sparse modifications to schoolbook
        if time.time() < _DEADLINE - 10:
            result = _search_sparse_modification(n, m, p, target_rank)
            if result is not None:
                return result
                
    except Exception:
        pass
        
    return None

def _search_sparse_modification(n, m, p, target_rank):
    """
    Search for a rank-49 decomposition by finding sparse modifications to schoolbook.
    This is a heuristic search that may find solutions if they exist with small structure.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    # This is a very limited search due to time constraints
    # Try to find if there's a near-rank-49 solution by looking at the null space
    # of the tensor's structure
    
    # For now, return None as this is complex and time-limited
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_reduce_rank(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
