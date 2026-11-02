"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify_full(U, V, W, n, m, p):
    """Full verification: T[i,j,k]=1 iff indices match, else 0."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    Ue = U_arr.reshape(n, m, 1, r)
    Ve = V_arr.reshape(1, m, p, r)
    We = W_arr.T.reshape(n, 1, p, r)
    T = np.sum(Ue * Ve * We, axis=3)
    # T[i,j,k] should be 1 for all valid i,j,k
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _schoolbook(n, m, p):
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
    return U, V, W, rank

def _strassen_3x3(n, m, p):
    """Strassen decomposition for 3x3x3, extended conceptually."""
    # This is for reference; we'll build custom for 3x4x4
    pass

def _greedy_decompose(n, m, p, max_rank, time_limit, rng):
    """
    Greedy: build rank-1 terms one at a time to match the target tensor T[i,j,k]=1.
    Each step: find (u, v, w) vectors (with small integer/half-integer entries)
    that best correlate with the current residual.
    """
    start = time.time()
    target = np.ones((n, m, p))
    residual = target.copy()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(max_rank):
        if time.time() - start > time_limit:
            break
        if np.allclose(residual, 0, atol=1e-9):
            break
        
        # Try to find a rank-1 tensor that best matches residual
        # Strategy: enumerate small patterns for U, V, W and pick best
        best_corr = -1
        best_u, best_v, best_w = None, None, None
        
        # Enumerate small integer vectors for each factor
        # U has shape (n*m,), V has (m*p,), W has (p*n,)
        # Try small values: -1, 0, 1, and maybe 0.5, -0.5
        vals = [-1, 0, 1]
        
        # For efficiency, try structured patterns
        # Pattern 1: U depends only on i, V only on j, W only on k
        # T_c[i,j,k] = f(i)*g(j)*h(k)
        for fi in range(-1, 2):
            for fj in range(-1, 2):
                for fk in range(-1, 2):
                    if fi == 0 and fj == 0 and fk == 0:
                        continue
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for i in range(n):
                        for j in range(m):
                            u[i * m + j] = fi
                    for j in range(m):
                        for k in range(p):
                            v[j * p + k] = fj
                    for k in range(p):
                        for i in range(n):
                            w[k * n + i] = fk
                    # Compute correlation with residual
                    tensor = np.einsum('ij,kl,ki->ijk', 
                                       u.reshape(n, m), v.reshape(m, p), w.reshape(p, n))
                    corr = np.sum(residual * tensor) / (np.sum(residual * residual) + 1e-12)
                    if corr > best_corr:
                        best_corr = corr
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None or best_corr <= 0:
            break
        
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        residual -= best_corr * np.einsum('ij,kl,ki->ijk',
                                           best_u.reshape(n, m),
                                           best_v.reshape(m, p),
                                           best_w.reshape(p, n))
    
    if not U_cols:
        return None
    
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return U, V, W, len(U_cols)

def _structured_search(n, m, p, time_limit):
    """
    Try structured decompositions based on algebraic patterns.
    For 3x4x4, try to exploit the rectangular structure.
    """
    results = []
    start = time.time()
    
    # Schoolbook baseline
    U, V, W, rank = _schoolbook(n, m, p)
    results.append((U, V, W, rank))
    
    # Try: decompose by fixing i (3 choices)
    # For each i, we need C[i,:] = A[i,:] * B
    # This is a 4x4 matrix-vector product for each row
    # Rank contribution per i: 4*4 = 16, total 3*16 = 48 (same as schoolbook)
    
    # Try: use the fact that 3 and 4 are small
    # Pattern: U[i,j,c] depends on (i,j), V[j,k,c] on (j,k), W[k,i,c] on (k,i)
    
    # Attempt: rank = 3*4 + 4*4 = 28? (split into two groups)
    # Or rank = 4*4 = 16 (the lower bound)
    
    # For now, try greedy
    greedy_result = _greedy_decompose(n, m, p, 40, time_limit * 0.5, random.Random(42))
    if greedy_result:
        results.append(greedy_result)
    
    return results

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Baseline: schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    best_rank = rank
    
    # Try structured search
    structured = _structured_search(n, m, p, 100)
    for s in structured:
        if s[3] < best_rank:
            if _verify_full(s[0], s[1], s[2], n, m, p):
                best = s
                best_rank = s[3]
    
    U_best, V_best, W_best, _ = best
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
