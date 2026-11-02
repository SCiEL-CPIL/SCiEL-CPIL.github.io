"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication as sparse dicts."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A10+A11)B00
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    # M3 = A00(B10-B11)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # M4 = A11(B01-B00)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # M5 = (A00+A01)B11
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using recursive Strassen (rank 49)."""
    terms_2x2 = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            for (bi, bj), cu in term1['U'].items():
                for (li, lj), cv in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i*4 + j][idx] += cu * cv
            for (bj, bk), cv in term1['V'].items():
                for (lj, lk), cw in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j*4 + k][idx] += cv * cw
            for (bk, bi), cw in term1['W'].items():
                for (lk, li), cu2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k*4 + i][idx] += cw * cu2
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _build_4x4x5_strassen():
    """Build 4x4x5: use 49-term Strassen for 4x4x4, then add 4 terms for 5th col."""
    n, m, p = DIMENSIONS
    base = _build_4x4x4_strassen()
    rank_base = 49
    rank = rank_base + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):
        for r in range(rank_base):
            V[j][r] = base["V"][j][r]
    for k in range(4 * n):
        for r in range(rank_base):
            W[k][r] = base["W"][k][r]
    
    # Add 4 terms for C[:,4] = A @ B[:,4]
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1
        for i in range(n):
            W[4*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x5_optimized():
    """Try to reduce rank by merging the 5th column terms with Strassen structure.
    
    Idea: The 5th column computation C[i,4] = sum_j A[i,j]*B[j,4] can potentially
    share structure with existing Strassen terms. Let's try a different decomposition
    where we treat the 4x4x5 problem more directly.
    
    Alternative: Use a 4x4x4 Strassen (49 terms) but modify some terms to also
    contribute to the 5th column, reducing the number of additional terms needed.
    """
    n, m, p = DIMENSIONS
    terms_2x2 = _strassen_2x2_terms()
    
    # Start with 49-term base
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    # Try rank 52: add only 3 terms instead of 4
    # This would require one of the 4 schoolbook terms to be absorbed
    # Let's try a different approach: use the structure more carefully
    
    # Actually, let's try to construct a 52-rank decomposition by
    # modifying one of the Strassen terms to also handle part of column 4
    
    # For now, return the 53-rank version and let optimization try to improve
    return _build_4x4x5_strassen()

def _try_direct_optimization(target_rank, n, m, p, max_iter=5000, seed=42):
    """Try to find a decomposition of given rank using Nelder-Mead optimization."""
    np.random.seed(seed)
    
    def _loss(params):
        U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U_p[i*m+j, r] * V_p[j*p+k, r] * W_p[k*n+i, r]
                    loss += (s - 1.0) ** 2
        return loss
    
    U_init = np.random.randn(n*m, target_rank) * 0.5
    V_init = np.random.randn(m*p, target_rank) * 0.5
    W_init = np.random.randn(p*n, target_rank) * 0.5
    params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    from scipy.optimize import minimize
    result = minimize(_loss, params, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    if result.fun < 1e-8:
        U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Round to half-integers and verify
        U_list = [[float(round(v * 2) / 2) for v in row] for row in U_opt]
        V_list = [[float(round(v * 2) / 2) for v in row] for row in V_opt]
        W_list = [[float(round(v * 2) / 2) for v in row] for row in W_opt]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _build_schoolbook():
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
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the 53-rank Strassen-based decomposition
    best_result = None
    try:
        result = _build_4x4x5_strassen()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
    except Exception:
        pass
    
    # Try to find lower rank through optimization
    if time.time() < deadline - 30:
        for target_rank in [52, 51, 50, 49]:
            if time.time() > deadline - 10:
                break
            try:
                result = _try_direct_optimization(target_rank, n, m, p, max_iter=2000, seed=target_rank*7)
                if result is not None:
                    best_result = result
                    break
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END

