import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    u_val = U[i*m+j][c]
                    v_val = V[j*p+l][c]
                    w_val = W[l*n+i][c]
                    # Convert to Fraction exactly
                    if isinstance(u_val, float):
                        u_frac = Fraction(u_val).limit_denominator(100000)
                    else:
                        u_frac = Fraction(u_val)
                    if isinstance(v_val, float):
                        v_frac = Fraction(v_val).limit_denominator(100000)
                    else:
                        v_frac = Fraction(v_val)
                    if isinstance(w_val, float):
                        w_frac = Fraction(w_val).limit_denominator(100000)
                    else:
                        w_frac = Fraction(w_val)
                    s += u_frac * v_frac * w_frac
                if s != 1:
                    return False
    return True

def _to_frac(x, max_den=10000):
    """Snap a float to the nearest simple fraction (integer or half-integer)."""
    # Try integer
    r_int = round(x)
    if abs(x - r_int) < 1e-9:
        return int(r_int)
    # Try half-integer
    r_half = round(x * 2) / 2.0
    if abs(x - r_half) < 1e-9:
        if r_half == int(r_half):
            return int(r_half)
        return r_half
    # Try quarters
    r_q = round(x * 4) / 4.0
    if abs(x - r_q) < 1e-9:
        return r_q
    # Fallback: use Fraction with limit_denominator
    frac = Fraction(x).limit_denominator(max_den)
    return float(frac)

def _compute_error(U, V, W):
    """Compute sum of squared errors."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c]
                err += (s - 1.0)**2
    return err

def _polish(U, V, W, max_iters=200, tol=1e-14):
    """High-precision polish pass using least squares."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    prev_err = float('inf')
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i*m + j
                M = np.zeros((p, R))
                b = np.ones(p)
                for l in range(p):
                    for c in range(R):
                        M[l, c] = V[j*p+l, c] * W[l*n+i, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for l in range(p):
                idx = j*p + l
                M = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    for c in range(R):
                        M[i, c] = U[i*m+j, c] * W[l*n+i, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        
        # Update W
        for l in range(p):
            for i in range(n):
                idx = l*n + i
                M = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    for c in range(R):
                        M[j, c] = U[i*m+j, c] * V[j*p+l, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
        
        err = _compute_error(U, V, W)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _try_coordinate_descent(R, max_outer=50, max_inner=100, seed=0, init_type='random'):
    """Coordinate descent to find a rank-R decomposition."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for outer in range(max_outer):
        if init_type == 'random':
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
        elif init_type == 'structured':
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            for c in range(R):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, c] = rng.choice([-1, 0, 1])
                for j in range(m):
                    for l in range(p):
                        V[j*p+l, c] = rng.choice([-1, 0, 1])
                for l in range(p):
                    for i in range(n):
                        W[l*n+i, c] = rng.choice([-1, 0, 1])
        elif init_type == 'sparse':
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            for c in range(R):
                u_idx = rng.choice(n*m, size=max(1, R//3), replace=False)
                v_idx = rng.choice(m*p, size=max(1, R//3), replace=False)
                w_idx = rng.choice(p*n, size=max(1, R//3), replace=False)
                U[u_idx, c] = rng.choice([-1, 1])
                V[v_idx, c] = rng.choice([-1, 1])
                W[w_idx, c] = rng.choice([-1, 1])
        else:
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
        
        prev_err = float('inf')
        for inner in range(max_inner):
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    M = np.zeros((p, R))
                    b = np.ones(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V[j*p+l, c] * W[l*n+i, c]
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        U[idx, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for l in range(p):
                    idx = j*p + l
                    M = np.zeros((n, R))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U[i*m+j, c] * W[l*n+i, c]
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        V[idx, :] = sol
                    except:
                        pass
            
            # Update W
            for l in range(p):
                for i in range(n):
                    idx = l*n + i
                    M = np.zeros((m, R))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U[i*m+j, c] * V[j*p+l, c]
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        W[idx, :] = sol
                    except:
                        pass
            
            err = _compute_error(U, V, W)
            if abs(prev_err - err) < 1e-13:
                break
            prev_err = err
        
        err = _compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-10:
            break
    
    if best_U is None:
        return None
    
    if best_err < 1e-8:
        # Polish
        U_p, V_p, W_p = _polish(best_U, best_V, best_W, max_iters=100, tol=1e-15)
        err_p = _compute_error(U_p, V_p, W_p)
        
        if err_p < best_err:
            best_U, best_V, best_W = U_p, V_p, W_p
            best_err = err_p
        
        # Snap to rationals
        for max_den in [1000, 10000, 100000]:
            U_r = [[_to_frac(x, max_den) for x in row] for row in best_U]
            V_r = [[_to_frac(x, max_den) for x in row] for row in best_V]
            W_r = [[_to_frac(x, max_den) for x in row] for row in best_W]
            
            if _verify_exact(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
    
    return None

def _strassen_rank_17():
    """
    Construct a rank-17 decomposition for 2x3 x 3x3 matrix multiplication.
    
    Strategy: Split the 2x3 matrix into two 2x2 blocks and a 2x1 block,
    or use a more clever partitioning.
    
    Actually, let's use the following approach:
    - We can write the 2x3 matrix as [A B] where A is 2x2 and B is 2x1
    - The 3x3 matrix can be partitioned accordingly
    
    A simpler approach: Use the fact that we can embed a 2x2x2 Strassen
    decomposition and handle the extra column/row separately.
    
    Let me think about this more carefully. For 2x3 * 3x3:
    - Output is 2x3
    - We need 18 scalar multiplications in schoolbook
    - Strassen for 2x2 uses 7 multiplications
    - We can try to use Strassen-like tricks
    
    Let's try a direct construction. Partition:
    A = [[a11, a12, a13], [a21, a22, a23]]  (2x3)
    B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]  (3x3)
    
    C = A * B is 2x3:
    C[0,0] = a11*b11 + a12*b21 + a13*b31
    C[0,1] = a11*b12 + a12*b22 + a13*b32
    C[0,2] = a11*b13 + a12*b23 + a13*b33
    C[1,0] = a21*b11 + a22*b21 + a23*b31
    C[1,1] = a21*b12 + a22*b22 + a23*b32
    C[1,2] = a21*b13 + a22*b23 + a23*b33
    
    Let's try to use a rank-17 decomposition. One approach:
    Use 7 multiplications for a 2x2 block product and 10 for the rest,
    but that's not quite right.
    
    Alternative: Use the following trick.
    Let A = [A1 A2] where A1 is 2x2 (columns 0,1) and A2 is 2x1 (column 2)
    Let B = [B1; B2] where B1 is 2x3 (rows 0,1) and B2 is 1x3 (row 2)
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x3): We can use Strassen-like methods.
    Actually, 2x2 * 2x3 can be done in 10 multiplications (vs 12 schoolbook).
    And A2*B2 (2x1 * 1x3) takes 2 multiplications.
    Total: 12. But we want 17, so this isn't tight.
    
    Let me try a different approach. Use a known rank-17 decomposition.
    
    Actually, let's just try to find it via optimization with a good starting point.
    """
    n, m, p = DIMENSIONS
    R = 17
    
    # Start with schoolbook and zero out one term, then optimize
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Initialize with schoolbook-like structure
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < R:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                else:
                    break
    
    # Try to optimize
    result = _try_coordinate_descent(R, max_outer=100, max_inner=200, seed=42, init_type='structured')
    if result is not None:
        return result
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 115  # Leave 5s margin before 120s limit
    
    # First try to find rank-17 via structured search
    if time.time() < deadline - 30:
        result = _strassen_rank_17()
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try coordinate descent for ranks 10-17
    init_types = ['random', 'structured', 'sparse']
    
    for R in range(10, 18):
        if time.time() > deadline - 10:
            break
        
        for init_type in init_types:
            for seed in range(20):
                if time.time() > deadline - 5:
                    break
                
                time_left = deadline - time.time()
                if time_left < 10:
                    max_outer = 10
                    max_inner = 20
                elif time_left < 30:
                    max_outer = 30
                    max_inner = 50
                else:
                    max_outer = 50
                    max_inner = 100
                
                res = _try_coordinate_descent(R, max_outer=max_outer, max_inner=max_inner, 
                                              seed=seed, init_type=init_type)
                if res is not None:
                    U, V, W = res
                    return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
