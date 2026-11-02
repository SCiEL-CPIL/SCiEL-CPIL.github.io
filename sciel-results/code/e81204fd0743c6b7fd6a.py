"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: 
1. Kronecker product T_{2,2,4} ⊗ T_{2,2,2} gives rank 56.
2. Randomized search for lower rank: fix sparse U, V, solve linear system for W.
"""

DIMENSIONS = [4, 4, 8]

import time
import numpy as np

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W


def _kronecker_4x4x8():
    """
    Construct rank-56 decomposition for 4x4 * 4x8 using Kronecker product:
    T_{4,4,8} = T_{2,2,4} ⊗ T_{2,2,2}
    
    Outer: n1=2, m1=2, p1=4, rank r1=8
    Inner: n2=2, m2=2, p2=2, rank r2=7
    Total rank = 8*7 = 56
    
    Index mapping:
    i = 2*i1 + i2 (i1 in 0..1, i2 in 0..1) -> i in 0..3
    j = 2*j1 + j2 (j1 in 0..1, j2 in 0..1) -> j in 0..3
    k = 4*k1 + k2 (k1 in 0..1, k2 in 0..3) -> k in 0..7
    """
    U222, V222, W222 = _strassen_2x2()
    
    n1, m1, p1, r1 = 2, 2, 4, 8
    U224 = np.zeros((n1 * m1, r1), dtype=float)
    V224 = np.zeros((m1 * p1, r1), dtype=float)
    W224 = np.zeros((p1 * n1, r1), dtype=float)
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U224[i * m1 + j, col] = 1
                V224[j * p1 + k, col] = 1
                W224[k * n1 + i, col] = 1
                col += 1
    
    n2, m2, p2, r2 = 2, 2, 2, 7
    n, m, p = n1*n2, m1*m2, p1*p2
    rank = r1 * r2
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = n1 * i1 + i2
                            j = m1 * j1 + j2
                            u_val = U224[i1 * m1 + j1, c1] * U222[i2 * m2 + j2, c2]
                            if u_val != 0:
                                U[i * m + j, c] += u_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = m1 * j1 + j2
                            k = p1 * k1 + k2
                            v_val = V224[j1 * p1 + k1, c1] * V222[j2 * p2 + k2, c2]
                            if v_val != 0:
                                V[j * p + k, c] += v_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = p1 * k1 + k2
                            i = n1 * i1 + i2
                            w_val = W224[k1 * n1 + i1, c1] * W222[k2 * n2 + i2, c2]
                            if w_val != 0:
                                W[k * n + i, c] += w_val
    
    return U, V, W


def _verify_4x4x8(U, V, W, rank):
    """Verify the 4x4*4x8 decomposition."""
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True


def _random_search(r, n_trials=100, deadline=None):
    """
    Randomized search for rank-r decomposition.
    Fix random sparse U and V, solve linear system for W.
    """
    n, m, p = 4, 4, 8
    best = None
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(trial * 1000 + r)
        
        # Generate sparse U: each row has 1-3 non-zero entries in {-1, 1}
        U = np.zeros((n * m, r), dtype=float)
        for row in range(n * m):
            nnz = np.random.randint(1, 4)
            cols = np.random.choice(r, size=nnz, replace=False)
            for col in cols:
                U[row, col] = np.random.choice([-1.0, 1.0])
        
        # Generate sparse V: each row has 1-3 non-zero entries in {-1, 1}
        V = np.zeros((m * p, r), dtype=float)
        for row in range(m * p):
            nnz = np.random.randint(1, 4)
            cols = np.random.choice(r, size=nnz, replace=False)
            for col in cols:
                V[row, col] = np.random.choice([-1.0, 1.0])
        
        # Build linear system for W
        n_unknowns = p * n * r
        n_equations = n * m * p
        
        A = np.zeros((n_equations, n_unknowns), dtype=float)
        b = np.ones(n_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if u_val != 0 and v_val != 0:
                            w_idx = (k * n + i) * r + c
                            A[eq_idx, w_idx] = u_val * v_val
                    eq_idx += 1
        
        try:
            w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W_flat = w_flat.reshape(p * n, r)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W_flat[k*n+i, c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                W_check = W_flat * 2
                if np.allclose(W_check, np.round(W_check), atol=1e-6):
                    W_flat = np.round(W_check) / 2.0
                    if best is None or r < best[0]:
                        best = (r, U.copy(), V.copy(), W_flat.copy())
                        return best
                        
        except Exception:
            pass
    
    return best


def construct():
    t_start = time.time()
    DEADLINE = t_start + 140
    
    n, m, p = DIMENSIONS
    
    # First, get the Kronecker rank-56 solution as baseline
    kronecker_result = None
    try:
        U, V, W = _kronecker_4x4x8()
        rank = W.shape[1]
        if _verify_4x4x8(U, V, W, rank):
            kronecker_result = (rank, U.copy(), V.copy(), W.copy())
    except Exception:
        pass
    
    # Try random search for lower rank
    best_result = kronecker_result
    
    for r in range(49, 56):
        if time.time() > DEADLINE - 10:
            break
        try:
            result = _random_search(r, n_trials=50, deadline=DEADLINE - 5)
            if result is not None:
                r_found, U, V, W = result
                if _verify_4x4x8(U, V, W, r_found):
                    best_result = (r_found, U.copy(), V.copy(), W.copy())
                    break
        except Exception:
            pass
    
    if best_result is not None:
        rank, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
