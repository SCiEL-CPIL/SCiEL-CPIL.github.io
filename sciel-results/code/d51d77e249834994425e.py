"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 x 2x2.
    Returns U (4x7), V (4x7), W (4x7) with W indexed by [k*2+i, r]."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
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

def _build_rank53():
    """Build the guaranteed rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    total_rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7

    # Last column: schoolbook for k=14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]

    return U, V, W

def _try_rank52(U53, V53, W53, deadline):
    """
    Try to find a rank-52 decomposition.
    
    Structural change: Instead of treating the full (2,2,15) problem as one giant
    1664-parameter optimization, we exploit the block structure.
    
    The rank-53 solution consists of 7 Strassen blocks (rank 7 each) and 1 schoolbook
    block (rank 4). We hypothesize that rank 52 is achievable by finding a rank-6
    decomposition for one of the 2x2x2 blocks, or by mixing the last schoolbook block
    with the adjacent Strassen block to save a term.
    
    However, M_{2,2,2} has rank 7, so we cannot reduce a single block.
    
    Alternative structural insight:
    The total rank is 53. The "waste" in the Strassen decomposition comes from the
    fact that we use 7 terms for 8 products. 
    If we look at the boundary between blocks, or the last block, can we do better?
    
    Let's try a different structural approach:
    Use 6 Strassen blocks (rank 42) for k=0..11.
    Remaining k=12, 13, 14.
    We need to decompose M_{2,2,3} for these last 3 columns.
    If we can find a rank-10 decomposition for M_{2,2,3}, total rank = 42 + 10 = 52.
    
    This is the key structural change: Instead of optimizing the whole 52x(4+30+30)
    matrix directly, we isolate the "tail" of the problem as M_{2,2,3} and search
    for a rank-10 solution for that specific sub-problem, which is much smaller
    (160 parameters vs 1664) and thus more tractable for global optimization.
    """
    n, m, p = DIMENSIONS
    
    # Target: 6 Strassen blocks + 1 M_{2,2,3} block of rank 10
    # Total rank: 6*7 + 10 = 42 + 10 = 52
    
    # First, verify we can find a rank-10 M_{2,2,3}
    U223, V223, W223 = _find_m223_rank10(deadline - 20)  # Leave 20s for assembly/verification
    
    if U223 is None:
        return None, None, None
    
    # Assemble the full rank-52 decomposition
    total_rank = 52
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Fill first 6 blocks with Strassen (k=0..11)
    col_offset = 0
    for b in range(6):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7
        
    # Fill last block with M_{2,2,3} (k=12,13,14)
    # U223 is 4x10, V223 is 6x10, W223 is 6x10
    # V223 indices: j*3+k_local where j in {0,1}, k_local in {0,1,2}
    # Global V indices: j*p + k_global where k_global in {12,13,14}
    # W223 indices: k_local*2+i where k_local in {0,1,2}, i in {0,1}
    # Global W indices: k_global*n + i
    
    U[:, col_offset:col_offset+10] = U223
    for j in range(m):
        for k_local in range(3):
            k_global = 12 + k_local
            V[j * p + k_global, col_offset:col_offset+10] = V223[j * 3 + k_local, :]
    for k_local in range(3):
        k_global = 12 + k_local
        for i in range(n):
            W[k_global * n + i, col_offset:col_offset+10] = W223[k_local * 2 + i, :]
            
    return U, V, W

def _find_m223_rank10(deadline):
    """
    Find a rank-10 decomposition for M_{2,2,3}.
    Uses a global optimization strategy with multiple random restarts.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    def residuals(params):
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    def objective(params):
        res = residuals(params)
        return np.sum(res**2)
        
    best_params = None
    best_obj = np.inf
    
    # Use a mix of structured and random initializations
    # Structured: Start from rank-11 (Strassen 7 + Schoolbook 4) and try to merge
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    U11[:, 0:7] = U_str
    for j in range(2):
        for k_local in range(2):
            V11[j*3+k_local, 0:7] = V_str[j*2+k_local, :]
    for k_local in range(2):
        for i in range(2):
            W11[k_local*2+i, 0:7] = W_str[k_local*2+i, :]
    U11[:, 7:11] = U_sb
    for j in range(2):
        V11[j*3+2, 7:11] = V_sb[j, :]
    for i in range(2):
        W11[2*2+i, 7:11] = W_sb[i, :]
        
    x0_structured = np.concatenate([U11.flatten(), V11.flatten(), W11.flatten()])
    
    seeds = [x0_structured]
    for seed in range(20):
        np.random.seed(seed)
        seeds.append(np.random.randn(160) * 0.5)
        
    for x0 in seeds:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-10:
                break
        except Exception:
            continue
            
    if best_params is None or best_obj > 1e-8:
        return None, None, None
        
    U = best_params[:40].reshape(4, 10)
    V = best_params[40:100].reshape(6, 10)
    W = best_params[100:160].reshape(6, 10)
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    res_snap = residuals(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]))
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap
        
    # If snapping fails, check if raw is close to half-integer
    if np.allclose(U*2, np.round(U*2), atol=1e-6) and \
       np.allclose(V*2, np.round(V*2), atol=1e-6) and \
       np.allclose(W*2, np.round(W*2), atol=1e-6):
        return U_snap, V_snap, W_snap

    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # Baseline: rank-53 solution
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
        U53, V53, W53 = _build_schoolbook(n, m, p)
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

    # Try to find a rank-52 solution
    if time.time() < deadline - 5:
        U52, V52, W52 = _try_rank52(U53, V53, W53, deadline)
        if U52 is not None:
            if _verify_decomposition(U52, V52, W52, n, m, p, tol=1e-6):
                return {"U": U52.tolist(), "V": V52.tolist(), "W": W52.tolist()}

    # Fallback: rank-53
    return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}
# EVOLVE-BLOCK-END
