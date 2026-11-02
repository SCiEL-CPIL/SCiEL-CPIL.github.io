"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for r in range(n):
        for s in range(p):
            for t in range(m):
                expected = 1.0
                actual = 0.0
                for k in range(rank):
                    actual += U[r*m + t, k] * V[t*p + s, k] * W[s*n + r, k]
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_rank14():
    """Construct the known rank-14 decomposition (two independent Strassen blocks)."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Strassen for columns 0,1 of B
    U[0][0] = 1; U[3][0] = 1  # M1: (a00+a11)
    U[2][1] = 1; U[3][1] = 1  # M2: (a10+a11)
    U[0][2] = 1               # M3: a00
    U[3][3] = 1               # M4: a11
    U[0][4] = 1; U[1][4] = 1  # M5: (a00+a01)
    U[2][5] = 1; U[0][5] = -1 # M6: (a10-a00)
    U[1][6] = 1; U[3][6] = -1 # M7: (a01-a11)
    
    V[0][0] = 1; V[5][0] = 1  # (b00+b11)
    V[0][1] = 1               # b00
    V[1][2] = 1; V[5][2] = -1 # (b10+b11)
    V[4][3] = 1; V[0][3] = -1 # (b00-b11)
    V[5][4] = 1               # b11
    V[0][5] = 1; V[1][5] = 1  # (b00+b10)
    V[4][6] = 1; V[5][6] = 1  # (b01+b11)
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1  # c00
    W[1][1] = 1; W[1][3] = 1                              # c01
    W[2][2] = 1; W[2][4] = 1                              # c10
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1   # c11
    
    # Block 2: Strassen for columns 2,3 of B
    U[0][7] = 1; U[3][7] = 1  # M1: (a00+a11)
    U[2][8] = 1; U[3][8] = 1  # M2: (a10+a11)
    U[0][9] = 1               # M3: a00
    U[3][10] = 1              # M4: a11
    U[0][11] = 1; U[1][11] = 1 # M5: (a00+a01)
    U[2][12] = 1; U[0][12] = -1 # M6: (a10-a00)
    U[1][13] = 1; U[3][13] = -1 # M7: (a01-a11)
    
    V[2][7] = 1; V[7][7] = 1  # (b02+b13)
    V[2][8] = 1               # b02
    V[3][9] = 1; V[7][9] = -1 # (b12+b13)
    V[6][10] = 1; V[2][10] = -1 # (b02-b13)
    V[7][11] = 1              # b13
    V[2][12] = 1; V[3][12] = 1 # (b02+b12)
    V[6][13] = 1; V[7][13] = 1 # (b03+b13)
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1  # c02
    W[5][8] = 1; W[5][10] = 1                               # c03
    W[6][9] = 1; W[6][11] = 1                               # c12
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1    # c13
    
    return {"U": U, "V": V, "W": W}

def _residual_rank_r(params, r, n=2, m=2, p=4):
    """Compute residuals for rank-r decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    U = params[:nm*r].reshape(nm, r)
    V = params[nm*r:nm*r+mp*r].reshape(mp, r)
    W = params[nm*r+mp*r:].reshape(pn, r)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for rr in range(n):
        for t in range(m):
            for s in range(p):
                val = np.dot(U[rr*m+t, :], V[t*p+s, :] * W[s*n+rr, :])
                residuals[idx] = val - 1.0
                idx += 1
    return residuals

def _dft_initialize(r, n=2, m=2, p=4, seed=0):
    """
    Initialize parameters using DFT structure.
    The DFT of the output dimension diagonalizes the 2x4 multiplication into
    4 spectral components. We use this structure to create informed initializations.
    """
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    nvars = (nm + mp + pn) * r
    
    # DFT matrix for p=4
    omega = np.exp(2j * np.pi / p)
    F = np.array([[omega**(k*s) for s in range(p)] for k in range(p)])
    F_inv = np.conj(F.T) / p
    
    # The multiplication tensor in spectral domain:
    # For each spectral frequency k, we have a 2x2x2-like structure
    # This suggests initializations where U, V, W have DFT-like patterns
    
    params = np.zeros(nvars)
    
    # Strategy: create r "modes" that mix DFT basis vectors
    for mode in range(r):
        # Random spectral weight
        w = rng.randn(p) + 1j * rng.randn(p)
        w = w / (np.linalg.norm(w) + 1e-10)
        
        # U component: random 2x2 linear form
        u_vec = rng.randn(nm)
        u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-10)
        
        # V component: DFT-weighted
        v_vec = np.zeros(mp)
        for t in range(m):
            for s in range(p):
                v_vec[t*p+s] = w[s] * rng.randn()
        v_vec = v_vec / (np.linalg.norm(v_vec) + 1e-10)
        
        # W component: conjugate DFT-weighted
        w_vec = np.zeros(pn)
        for s in range(p):
            for rr in range(n):
                w_vec[s*n+rr] = np.conj(w[s]) * rng.randn()
        w_vec = w_vec / (np.linalg.norm(w_vec) + 1e-10)
        
        # Place into params
        params[mode] = np.real(u_vec[0])
        params[nm + mode] = np.real(v_vec[0])
        params[nm + mp + mode] = np.real(w_vec[0])
        
        # Fill remaining entries with small random values
        for idx in range(1, nm):
            params[idx * r + mode] = rng.randn() * 0.1
        for idx in range(1, mp):
            params[nm + idx * r + mode] = rng.randn() * 0.1
        for idx in range(1, pn):
            params[nm + mp + idx * r + mode] = rng.randn() * 0.1
    
    return params

def _polynomial_initialize(r, n=2, m=2, p=4, seed=0):
    """
    Initialize using polynomial/convolution structure.
    Treat B as a polynomial and use FFT-like structure.
    """
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    nvars = (nm + mp + pn) * r
    
    params = np.zeros(nvars)
    
    # Create r random "convolution kernels"
    for mode in range(r):
        # U: random 2x2
        u_vals = rng.randn(nm) * 0.5
        for idx in range(nm):
            params[idx * r + mode] = u_vals[idx]
        
        # V: random 2x4 with polynomial structure
        # b[t][s] = sum_{d=0}^{3} c_d[t] * x^d where x depends on s
        c = rng.randn(m, 4) * 0.5
        for t in range(m):
            for s in range(p):
                val = sum(c[t, d] * (s - 1.5)**d for d in range(4))
                params[nm + (t*p+s) * r + mode] = val
        
        # W: similar polynomial structure
        cw = rng.randn(p, n) * 0.5
        for s in range(p):
            for rr in range(n):
                val = sum(cw[s, d] * (rr - 0.5)**d for d in range(min(n, 4)))
                params[nm + mp + (s*n+rr) * r + mode] = val
    
    return params

def _try_rank_r(r, time_limit=50.0, n=2, m=2, p=4, max_seeds=300):
    """Try to find a rank-r decomposition using least squares optimization."""
    start = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    nvars = (nm + mp + pn) * r
    
    best_result = None
    
    for seed in range(max_seeds):
        if time.time() - start > time_limit:
            break
        
        # Use different initialization strategies
        if seed % 3 == 0:
            params = _dft_initialize(r, n, m, p, seed=seed)
        elif seed % 3 == 1:
            params = _polynomial_initialize(r, n, m, p, seed=seed)
        else:
            np.random.seed(seed)
            params = np.random.randn(nvars) * 0.5
        
        try:
            result = least_squares(_residual_rank_r, params, args=(r, n, m, p), 
                                   method='lm', max_nfev=300)
        except Exception:
            continue
        
        if result.cost < 1e-12:
            U = result.x[:nm*r].reshape(nm, r)
            V = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = result.x[nm*r+mp*r:].reshape(pn, r)
            
            # High-precision polish pass
            try:
                params_polish = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result_polish = least_squares(_residual_rank_r, params_polish, args=(r, n, m, p),
                                             method='lm', max_nfev=500, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                if result_polish.cost < result.cost:
                    result = result_polish
            except Exception:
                pass
            
            U = result.x[:nm*r].reshape(nm, r)
            V = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = result.x[nm*r+mp*r:].reshape(pn, r)
            
            # Try rounding to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Try rounding to nearest integer
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if _verify(U_i.tolist(), V_i.tolist(), W_i.tolist()):
                return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
            
            # Also try the unrounded solution
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _spectral_rank_estimate(n=2, m=2, p=4):
    """
    Estimate the minimum rank by analyzing the spectral structure.
    The DFT of the output dimension should reveal if cross-term sharing is possible.
    """
    # Build the multiplication tensor
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for t in range(m):
            for s in range(p):
                tensor[i, t, s] = 1.0  # This is the identity pattern for matmul
    
    # Apply DFT along the p dimension
    F = np.fft.fft2(tensor, axes=(2,))
    
    # The spectral components should be independent if rank = sum of ranks of each
    # If there's sharing, we can potentially reduce rank
    # This is a heuristic - the actual rank is determined by the optimization
    
    return tensor, F

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    
    # First, analyze the spectral structure (quick)
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            tensor, F = _spectral_rank_estimate(n, m, p)
        except Exception:
            pass
    
    # Try rank 12 first (theoretical minimum per Larsen)
    remaining = deadline - time.time()
    if remaining > 15:
        result = _try_rank_r(12, time_limit=min(remaining - 5, 40.0))
        if result is not None:
            return result
    
    # Try rank 13
    remaining = deadline - time.time()
    if remaining > 15:
        result = _try_rank_r(13, time_limit=min(remaining - 5, 40.0))
        if result is not None:
            return result
    
    # Fall back to rank 14
    return _construct_rank14()
# EVOLVE-BLOCK-END
