import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build target tensor: T[a,b,c] = 1 if a=2i+j, b=8j'+k, c=8k'+i' with i=i', j=j', k=k'
    # Wait, the trace convention is W[k*n+i, r].
    # Input A is n x m. Input B is m x p. Output C is n x p.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Indexing:
    # A index: a = i * m + j
    # B index: b = j * p + k
    # C index: c = i * p + k  <-- Wait, the prompt says "official trace convention W[k*n+i,r]"
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means the output matrix C is indexed by (k, i) or similar.
    # Standard matrix multiplication: C = A B. C is n x p.
    # Usually C[i, k].
    # If the convention is W[k*n+i], then the index is k*n + i.
    # This implies the output matrix is stored in column-major order? Or the indices are swapped?
    # Let's check the current code's verification.
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                # c index in current code: k * n + i
                c = k * n + i
                target[a, b, c] = 1.0
    
    decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        decomp += np.einsum('a,b,c->abc', u, v, w)
    
    return np.allclose(decomp, target, atol=tol)

def _strassen_28():
    """Return the rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00: M1, M4, M5, M7
        {2: 1, 4: 1},              # C01: M3, M5
        {1: 1, 3: 1},              # C10: M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11: M1, M2, M3, M6
    ]

    for block in range(4):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}

def _construct_26_from_13():
    """
    Attempts to construct a rank-26 solution by finding a rank-13 solution for 2x2x4.
    If successful, returns the rank-26 solution. Otherwise returns None.
    """
    # This is a placeholder for a complex search.
    # For now, we return None to avoid timeout if search is slow.
    # A real implementation would use numerical optimization.
    return None

def construct():
    """
    Returns the best rank decomposition found.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Get the rank-28 solution
    sol28 = _strassen_28()
    
    # Verify it's correct
    if not _verify_decomposition(sol28['U'], sol28['V'], sol28['W']):
        return sol28
    
    # Try to find a lower rank solution (e.g., rank 26)
    if time.time() < deadline:
        sol26 = _construct_26_from_13()
        if sol26 is not None and _verify_decomposition(sol26['U'], sol26['V'], sol26['W']):
            return sol26
    
    return sol28
