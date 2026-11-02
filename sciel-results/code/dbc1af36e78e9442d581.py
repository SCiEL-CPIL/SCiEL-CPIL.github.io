"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 x 2x2.
    Returns U (4x7), V (4x7), W (4x7) with W indexed by [k*2+i, r]."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _verify_225(U, V, W, tol=1e-8):
    """Verify 2x2x5 decomposition: sum_r U[i*2+j,r]*V[j*5+k,r]*W[k*2+i,r] == 1."""
    n, m, p = 2, 2, 5
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _residuals_225(U, V, W):
    """Residuals for (2,2,5) tensor: 20 equations."""
    n, m, p = 2, 2, 5
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
                idx += 1
    return res

def _try_m225_rank15_structured(deadline):
    """
    Attempt to find a rank-15 decomposition for M_{2,2,5} using structured search.
    Strategy: Use a parametric family where U has a specific block structure,
    then solve for V and W via least squares or optimization.
    Returns (U, V, W) as numpy arrays or None.
    """
    start = time.time()
    
    def _snap_and_verify(U, V, W, tol=1e-6):
        """Try snapping to half-integers, integers, and quarters."""
        for mult in [2, 1, 4]:
            U_s = np.round(U * mult) / mult
            V_s = np.round(V * mult) / mult
            W_s = np.round(W * mult) / mult
            res = _residuals_225(U_s, V_s, W_s)
            if np.max(np.abs(res)) < tol:
                return U_s, V_s, W_s
        return None
    
    # Strategy 1: Fix U to a structured form, solve for V, W
    # The 2x2x5 tensor: for each k in 0..4, we have a 2x2 matrix product
    # Key insight: if we can find U such that the "column slices" have good properties
    
    # Approach: Use random U with structure, then solve linear system for V*W products
    # For each r, U[:,r] * V[:,r] * W[:,r] must sum to the tensor
    
    # More direct: treat as bilinear in (V, W) given U
    # For fixed U, the constraint is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # This is still nonlinear in V, W jointly
    
    # Better: Use the fact that we can parameterize and use scipy with good init
    
    # Try multiple structured initializations
    best_result = None
    best_obj = np.inf
    
    # Initialization strategy A: Start from 3x(2x2x2) Strassen + 1x(2x2x1) schoolbook = 3*7+4=25, then merge
    # Actually, let's try direct optimization with diverse starts
    
    n, m, p = 2, 2, 5
    rank = 15
    nU, nV, nW = n*m, m*p, p*n  # 4, 10, 10
    total_params = nU*rank + nV*rank + nW*rank  # 60+150+150=360
    
    def objective(params):
        U = params[:nU*rank].reshape(nU, rank)
        V = params[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
        W = params[nU*rank+nV*rank:].reshape(nW, rank)
        res = _residuals_225(U, V, W)
        return np.sum(res**2)
    
    # Try diverse initializations
    inits = []
    
    # Init 1: Small random
    np.random.seed(42)
    inits.append(np.random.randn(total_params) * 0.3)
    
    # Init 2: Sparse random
    np.random.seed(123)
    x0 = np.random.randn(total_params)
    mask = np.random.rand(total_params) < 0.15
    x0[~mask] = 0
    x0[mask] *= 1.5
    inits.append(x0)
    
    # Init 3: Structured - U has identity-like pattern in first 4 cols
    np.random.seed(777)
    x0 = np.zeros(total_params)
    U_part = x0[:nU*rank].reshape(nU, rank)
    # Set first 4 columns of U to identity
    for i in range(min(4, nU)):
        U_part[i, i] = 1.0
    # Random for rest
    V_part = x0[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
    W_part = x0[nU*rank+nV*rank:].reshape(nW, rank)
    V_part += np.random.randn(nV, rank) * 0.5
    W_part += np.random.randn(nW, rank) * 0.5
    inits.append(x0)
    
    # Init 4: Based on scaling Strassen
    U_str, V_str, W_str = _build_strassen_2x2x2()
    # Extend to 2x2x5 by repeating/adjusting
    x0 = np.zeros(total_params)
    U_part = x0[:nU*rank].reshape(nU, rank)
    V_part = x0[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
    W_part = x0[nU*rank+nV*rank:].reshape(nW, rank)
    # Fill first 7 columns with Strassen-like pattern
    U_part[:, :7] = U_str
    # For V: Strassen V is 4x7, we need 10x15. Map j*2+k -> j*5+k'
    for j in range(2):
        for k in range(2):
            V_part[j*5+k, :7] = V_str[j*2+k, :]
            V_part[j*5+k+2, :7] = 0  # k=2 is new
    # For W: Strassen W is 4x7, we need 10x15. Map k*2+i -> k*2+i
    for k in range(2):
        for i in range(2):
            W_part[k*2+i, :7] = W_str[k*2+i, :]
    # Add random perturbation for remaining columns
    U_part[:, 7:] += np.random.randn(nU, 8) * 0.3
    V_part[:, 7:] += np.random.randn(nV, 8) * 0.3
    W_part[:, 7:] += np.random.randn(nW, 8) * 0.3
    inits.append(x0)
    
    # Init 5: All-ones scaled
    np.random.seed(999)
    x0 = np.ones(total_params) * 0.2
    x0 += np.random.randn(total_params) * 0.1
    inits.append(x0)
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
                
            if best_obj < 1e-12:
                break
                
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    U = best_params[:nU*rank].reshape(nU, rank)
    V = best_params[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
    W = best_params[nU*rank+nV*rank:].reshape(nW, rank)
    
    # Snap and verify
    result = _snap_and_verify(U, V, W)
    if result is not None:
        return result
    
    # If snapping failed, try with looser tolerance
    res = _residuals_225(U, V, W)
    if np.max(np.abs(res)) < 1e-4:
        # Try more aggressive snapping
        for mult in [2, 1, 4, 8]:
            U_s = np.round(U * mult) / mult
            V_s = np.round(V * mult) / mult
            W_s = np.round(W * mult) / mult
            res_s = _residuals_225(U_s, V_s, W_s)
            if np.max(np.abs(res_s)) < 1e-6:
                return U_s, V_s, W_s
    
    return None

def _try_m225_rank15_fast(deadline):
    """
    Faster attempt: Use a reduced parameterization.
    Key insight: The 2x2x5 tensor has symmetries. We can fix some degrees of freedom.
    Returns (U, V, W) as numpy arrays or None.
    """
    start = time.time()
    
    # Alternative approach: Use the fact that M_{2,2,5} can be decomposed as
    # a sum of rank-1 tensors. Try a specific algebraic construction.
    
    # Known result: rank(M_{2,2,5}) <= 15 is conjectured but not proven in general.
    # Let's try a different tactic: optimize with more iterations but fewer params
    # by fixing U to be in a low-dimensional subspace.
    
    # Parametrize U as U = A * B where A is 4x8, B is 8x15 (rank at most 8)
    # This reduces U params from 60 to 4*8+8*15=152, but adds structure
    
    # Actually, let's try: fix U to have only 5 non-zero columns (one per k-slice)
    # U[i,j,r] = 0 unless r corresponds to a specific k
    # This gives a block-diagonal structure
    
    # For each k in 0..4, assign 3 columns of U to that k
    # U[i,j,3k], U[i,j,3k+1], U[i,j,3k+2] are non-zero
    # Then V[j,k,r] and W[k,i,r] must be zero for r not in {3k,3k+1,3k+2}
    
    # This reduces the problem to 5 independent 2x2x1 problems, each of rank 3
    # But 2x2x1 has rank 4, so this won't work directly.
    
    # Let's try a different block structure: 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    # Rank: 7 + 7 + 4 = 18, not 15.
    
    # Try: 2x2x5 with shared structure across k
    # U[i,j,r] depends on r mod 5 in a periodic way
    
    # I'll try a direct optimization with a very tight budget
    n, m, p = 2, 2, 5
    rank = 15
    nU, nV, nW = n*m, m*p, p*n
    total_params = nU*rank + nV*rank + nW*rank
    
    def objective(params):
        U = params[:nU*rank].reshape(nU, rank)
        V = params[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
        W = params[nU*rank+nV*rank:].reshape(nW, rank)
        res = _residuals_225(U, V, W)
        return np.sum(res**2)
    
    # Use a single good initialization with more iterations
    np.random.seed(2024)
    x0 = np.random.randn(total_params) * 0.25
    
    # Add a structured component
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_part = x0[:nU*rank].reshape(nU, rank)
    V_part = x0[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
    W_part = x0[nU*rank+nV*rank:].reshape(nW, rank)
    
    # Blend Strassen into first 7 columns
    U_part[:, :7] = 0.5 * U_str + 0.5 * U_part[:, :7]
    for j in range(2):
        for k in range(2):
            V_part[j*5+k, :7] = 0.5 * V_str[j*2+k, :] + 0.5 * V_part[j*5+k, :7]
    for k in range(2):
        for i in range(2):
            W_part[k*2+i, :7] = 0.5 * W_str[k*2+i, :] + 0.5 * W_part[k*2+i, :7]
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 20000})
        
        if res.fun < 1e-10:
            U = res.x[:nU*rank].reshape(nU, rank)
            V = res.x[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
            W = res.x[nU*rank+nV*rank:].reshape(nW, rank)
            
            for mult in [2, 1, 4]:
                U_s = np.round(U * mult) / mult
                V_s = np.round(V * mult) / mult
                W_s = np.round(W * mult) / mult
                res_s = _residuals_225(U_s, V_s, W_s)
                if np.max(np.abs(res_s)) < 1e-6:
                    return U_s, V_s, W_s
            
    except Exception:
        pass
    
    return None

def _build_rank53():
    """Build the guaranteed rank-53 decomposition."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_school = [[1,0,1,0],[0,1,0,1]]
    W_school = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    return {"U": U, "V": V, "W": W}

def _build_rank50_m225(U_m225, V_m225, W_m225):
    """
    Build rank-50 decomposition:
    - Columns 0-34: five 2x2x2 Strassen blocks (rank 35)
    - Columns 35-49: one 2x2x5 block (rank 15)
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Place 5 Strassen blocks for columns 0-34
    col_offset = 0
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = int(U_strassen[i * 2 + j][r])
            for j in range(2):
                V[j * p + k0][c] = int(V_strassen[j * 2][r])
                V[j * p + k1][c] = int(V_strassen[j * 2 + 1][r])
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = int(W_strassen[k * 2 + i][r])
        col_offset += 7
    
    # Place 2x2x5 block for columns 35-49
    k0 = 10
    for r in range(15):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = int(U_m225[i * 2 + j][r])
        for j in range(2):
            for k in range(5):
                V[j * p + k0 + k][c] = int(V_m225[j * 5 + k][r])
        for k in range(5):
            for i in range(2):
                W[(k0 + k) * n + i][c] = int(W_m225[k * 2 + i][r])
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try to find rank-15 M_{2,2,5} decomposition with structured search
    m225_result = _try_m225_rank15_structured(deadline=start_time + 60)
    
    if m225_result is None and time.time() < deadline:
        # Try faster variant
        m225_result = _try_m225_rank15_fast(deadline=start_time + 100)
    
    if m225_result is not None and time.time() < deadline:
        U_m225, V_m225, W_m225 = m225_result
        # Verify before building
        if _verify_225(U_m225, V_m225, W_m225, tol=1e-6):
            return _build_rank50_m225(U_m225, V_m225, W_m225)
    
    # Fallback to guaranteed rank-53
    return _build_rank53()
# EVOLVE-BLOCK-END
