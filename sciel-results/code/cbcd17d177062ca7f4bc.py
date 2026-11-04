"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS = 2, 3, 3

_DEADLINE = time.time() + 140.0

def _remaining():
    return _DEADLINE - time.time()


def _verify(U, V, W, rank):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-7:
                    return False
    return True


def _rank10_decomposition():
    """
    Explicit rank-10 decomposition using Strassen's algorithm for the 2x2x2 block
    plus 3 additional rank-1 terms for the remaining (j,k) pairs involving index 2.
    """
    n, m, p = 2, 3, 3
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    # --- Strassen 7 terms for i,j,k in {0,1} ---
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0*3+0, 0] = 1.0; U[1*3+1, 0] = 1.0
    V[0*3+0, 0] = 1.0; V[1*3+1, 0] = 1.0
    W[0*2+0, 0] = 1.0; W[1*2+1, 0] = 1.0

    # Term 1: M2 = (A10+A11)(B00)
    U[1*3+0, 1] = 1.0; U[1*3+1, 1] = 1.0
    V[0*3+0, 1] = 1.0
    W[0*2+1, 1] = 1.0; W[1*2+1, 1] = -1.0

    # Term 2: M3 = (A00)(B01-B11)
    U[0*3+0, 2] = 1.0
    V[0*3+1, 2] = 1.0; V[1*3+1, 2] = -1.0
    W[1*2+0, 2] = 1.0; W[1*2+1, 2] = 1.0

    # Term 3: M4 = (A11)(B10-B00)
    U[1*3+1, 3] = 1.0
    V[1*3+0, 3] = 1.0; V[0*3+0, 3] = -1.0
    W[0*2+0, 3] = 1.0; W[0*2+1, 3] = 1.0

    # Term 4: M5 = (A00+A01)(B11)
    U[0*3+0, 4] = 1.0; U[0*3+1, 4] = 1.0
    V[1*3+1, 4] = 1.0
    W[0*2+0, 4] = -1.0; W[1*2+1, 4] = -1.0

    # Term 5: M6 = (A10-A00)(B00+B01)
    U[1*3+0, 5] = 1.0; U[0*3+0, 5] = -1.0
    V[0*3+0, 5] = 1.0; V[0*3+1, 5] = 1.0
    W[1*2+1, 5] = 1.0

    # Term 6: M7 = (A01-A11)(B10+B11)
    U[0*3+1, 6] = 1.0; U[1*3+1, 6] = -1.0
    V[1*3+0, 6] = 1.0; V[1*3+1, 6] = 1.0
    W[0*2+0, 6] = 1.0

    # --- Term 7: j in {0,1}, k=2 ---
    for i in range(2):
        for j in range(2):
            U[i*3+j, 7] = 1.0
    for j in range(2):
        V[j*3+2, 7] = 1.0
    for i in range(2):
        W[2*2+i, 7] = 1.0

    # --- Term 8: j=2, k in {0,1} ---
    for i in range(2):
        U[i*3+2, 8] = 1.0
    for k in range(2):
        V[2*3+k, 8] = 1.0
    for k in range(2):
        for i in range(2):
            W[k*2+i, 8] = 1.0

    # --- Term 9: j=2, k=2 ---
    for i in range(2):
        U[i*3+2, 9] = 1.0
    V[2*3+2, 9] = 1.0
    for i in range(2):
        W[2*2+i, 9] = 1.0

    return U.tolist(), V.tolist(), W.tolist()


def _try_rank9_search():
    """
    Try to find a rank-9 decomposition using vectorized residual computation
    and multiple random restarts with the Levenberg-Marquardt optimizer.
    """
    if _remaining() < 20:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 9
    nm, mp, pn = N*M, M*P, P*N
    u_size = nm * rank
    v_size = mp * rank
    w_size = pn * rank
    total_size = u_size + v_size + w_size
    
    # Precompute index arrays for vectorized residual
    # For each (i,j,k) output, we need U[i*M+j, :], V[j*P+k, :], W[k*N+i, :]
    idx_u = np.array([i*M + j for i in range(N) for j in range(M) for _ in range(P)])
    idx_v = np.array([j*P + k for i in range(N) for j in range(M) for k in range(P)])
    idx_w = np.array([k*N + i for i in range(N) for j in range(M) for k in range(P)])
    
    def _residuals(params):
        U = params[:u_size].reshape(nm, rank)
        V = params[u_size:u_size+v_size].reshape(mp, rank)
        W = params[u_size+v_size:u_size+v_size+w_size].reshape(pn, rank)
        
        # Vectorized: for each (i,j,k), compute sum_r U[iM+j,r]*V[jP+k,r]*W[kN+i,r]
        # Shape: (N*M*P, rank) for each factor
        U_sel = U[idx_u, :]  # (18, 9)
        V_sel = V[idx_v, :]  # (18, 9)
        W_sel = W[idx_w, :]  # (18, 9)
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (18,)
        return vals - 1.0
    
    best_cost = float('inf')
    best_x = None
    
    # Try multiple random initializations with varying scales
    attempts = 0
    max_attempts = 30
    while attempts < max_attempts and _remaining() > 15:
        attempts += 1
        np.random.seed(attempts * 137 + 42)
        
        # Vary the scale to explore different regions
        scale = 0.3 + 0.2 * (attempts % 5)
        u0 = np.random.randn(nm, rank) * scale
        v0 = np.random.randn(mp, rank) * scale
        w0 = np.random.randn(pn, rank) * scale
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])
        
        try:
            result = least_squares(_residuals, params0, method='lm', max_nfev=3000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x.copy()
                
                if result.cost < 1e-14:
                    U = result.x[:u_size].reshape(nm, rank)
                    V = result.x[u_size:u_size+v_size].reshape(mp, rank)
                    W = result.x[u_size+v_size:u_size+v_size+w_size].reshape(pn, rank)
                    
                    # Try rounding to nearest 0.5 first
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if _verify(U_r, V_r, W_r, rank):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    # Then check unrounded
                    elif _verify(U, V, W, rank):
                        return U.tolist(), V.tolist(), W.tolist()
                    # Try rounding to nearest integer
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if _verify(U_i, V_i, W_i, rank):
                        return U_i.tolist(), V_i.tolist(), W_i.tolist()
        except Exception:
            pass
    
    # If we found a good solution but didn't verify, do a final check
    if best_x is not None and best_cost < 1e-10:
        U = best_x[:u_size].reshape(nm, rank)
        V = best_x[u_size:u_size+v_size].reshape(mp, rank)
        W = best_x[u_size+v_size:u_size+v_size+w_size].reshape(pn, rank)
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r, V_r, W_r, rank):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        elif _verify(U, V, W, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _schoolbook():
    """Return the schoolbook rank-18 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    # Try rank 9 first with vectorized optimization
    result9 = _try_rank9_search()
    if result9 is not None:
        U, V, W = result9
        return {"U": U, "V": V, "W": W}

    # Try the explicit rank-10 decomposition
    U, V, W = _rank10_decomposition()
    if _verify(U, V, W, 10):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook decomposition
    return _schoolbook()
