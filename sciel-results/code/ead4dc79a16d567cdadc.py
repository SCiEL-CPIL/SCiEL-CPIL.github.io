import numpy as np
import time
import scipy.optimize
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    rank = len(U[0])
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            # Expected: 1 if i==ip and j==jp and k==kp, else 0
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            
                            # Computed: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def _round_to_half_integer(mat):
    """Round matrix entries to nearest half-integer."""
    return np.round(np.array(mat) * 2) / 2.0

def _construct_tensor(n, m, p):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_W, dim_U, dim_V), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0
    return T

def _try_rank_optimization(n, m, p, r, T, time_limit=10):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Total variables: dim_U*r + dim_V*r + dim_W*r
    num_vars = (dim_U + dim_V + dim_W) * r
    
    def unpack(x):
        U = x[:dim_U*r].reshape(dim_U, r)
        V = x[dim_U*r:(dim_U+dim_V)*r].reshape(dim_V, r)
        W = x[(dim_U+dim_V)*r:].reshape(dim_W, r)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        # Compute T_rec
        # T_rec[a, b, c] = sum_r W[a, r] * U[b, r] * V[c, r]
        T_rec = np.einsum('ar,br,cr->abc', W, U, V)
        return (T_rec - T).flatten()

    # Try a few random initializations
    start_time = time.time()
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(attempt * 100 + r)
        x0 = np.random.randn(num_vars) * 0.5
        
        try:
            # Use least_squares which is robust
            # Method 'trf' is good for general problems
            result = scipy.optimize.least_squares(
                residuals, x0, method='trf', 
                max_nfev=1000, # Limit function evaluations to prevent hanging
                xtol=1e-8, ftol=1e-8, gtol=1e-8
            )
            
            x_opt = result.x
            U, V, W = unpack(x_opt)
            
            # Snap to half-integers
            U_snap = _round_to_half_integer(U)
            V_snap = _round_to_half_integer(V)
            W_snap = _round_to_half_integer(W)
            
            # Verify exactness of snapped solution
            if _verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
            # If snapping failed, check if the float solution is very close to half-integers
            # and verify the float solution directly (though task requires half-integers)
            # If float solution is exact but not half-integer, it's invalid for the contract.
            # So we must rely on snapping.
            
        except Exception as e:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_schoolbook = n * m * p  # 24
    
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    T = _construct_tensor(n, m, p)
    
    best_result = None
    
    # Try ranks from 18 to 23
    # 18 is the theoretical minimum for 2x3x4
    for r in range(18, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time based on rank likelihood
        # Lower ranks are harder, give them more time? Or less?
        # Let's give 15s to 18, 10s to 19, 5s to others.
        if r == 18:
            t_lim = 20
        elif r == 19:
            t_lim = 15
        elif r == 20:
            t_lim = 10
        else:
            t_lim = 5
            
        if time.time() - start_time + t_lim > time_limit:
            t_lim = max(1, time_limit - (time.time() - start_time))
            
        res = _try_rank_optimization(n, m, p, r, T, time_limit=t_lim)
        if res is not None:
            best_result = res
            break # Found a valid decomposition, return it (lowest rank first)
            
    # If no low rank found, return schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
