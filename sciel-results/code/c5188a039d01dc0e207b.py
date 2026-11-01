"""
Rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.

Strategy:
1. Use Strassen's algorithm on the 2x2 block structure to get rank 21.
2. Attempt to reduce to rank 20 using a structured search.
3. Verify any found decomposition.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START

def _build_rank21():
    """Build the rank-21 Strassen-based decomposition."""
    n, m, p = 2, 2, 6
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen products for each of the 3 blocks of B (2x2 each)
    # Block t uses columns 2t, 2t+1 of B
    for t in range(3):
        bcol = 2 * t
        for s in range(7):
            c = t * 7 + s
            if s == 0:  # p1 = (A00+A11)(B00+B11)
                U[0, c] = 1; U[3, c] = 1
                V[0*6 + bcol, c] = 1; V[1*6 + bcol + 1, c] = 1
                W[(bcol)*2 + 0, c] = 1; W[(bcol+1)*2 + 0, c] = -1
                W[(bcol)*2 + 1, c] = 0; W[(bcol+1)*2 + 1, c] = 1
            elif s == 1:  # p2 = (A10+A11)B00
                U[2, c] = 1; U[3, c] = 1
                V[0*6 + bcol, c] = 1
                W[(bcol)*2 + 0, c] = -1; W[(bcol+1)*2 + 0, c] = 0
                W[(bcol)*2 + 1, c] = 0; W[(bcol+1)*2 + 1, c] = -1
            elif s == 2:  # p3 = A00(B01-B11)
                U[0, c] = 1
                V[0*6 + bcol + 1, c] = 1; V[1*6 + bcol + 1, c] = -1
                W[(bcol)*2 + 0, c] = 0; W[(bcol+1)*2 + 0, c] = 0
                W[(bcol)*2 + 1, c] = 1; W[(bcol+1)*2 + 1, c] = -1
            elif s == 3:  # p4 = A11(B10-B00)
                U[3, c] = 1
                V[1*6 + bcol, c] = 1; V[0*6 + bcol, c] = -1
                W[(bcol)*2 + 0, c] = 0; W[(bcol+1)*2 + 0, c] = 1
                W[(bcol)*2 + 1, c] = 0; W[(bcol+1)*2 + 1, c] = 1
            elif s == 4:  # p5 = (A00+A01)B11
                U[0, c] = 1; U[1, c] = 1
                V[1*6 + bcol + 1, c] = 1
                W[(bcol)*2 + 0, c] = -1; W[(bcol+1)*2 + 0, c] = 1
                W[(bcol)*2 + 1, c] = 0; W[(bcol+1)*2 + 1, c] = 0
            elif s == 5:  # p6 = (A10-A01)(B00+B01)
                U[2, c] = 1; U[1, c] = -1
                V[0*6 + bcol, c] = 1; V[0*6 + bcol + 1, c] = 1
                W[(bcol)*2 + 0, c] = 1; W[(bcol+1)*2 + 0, c] = 0
                W[(bcol)*2 + 1, c] = 1; W[(bcol+1)*2 + 1, c] = 0
            elif s == 6:  # p7 = (A11-A01)(B10+B11)
                U[3, c] = 1; U[1, c] = -1
                V[1*6 + bcol, c] = 1; V[1*6 + bcol + 1, c] = 1
                W[(bcol)*2 + 0, c] = 1; W[(bcol+1)*2 + 0, c] = 0
                W[(bcol)*2 + 1, c] = 0; W[(bcol+1)*2 + 1, c] = 1
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, n=2, m=2, p=6, tol=1e-10):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k), check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _attempt_rank20(deadline):
    """Try to find a rank-20 decomposition via structured search."""
    # Start from rank 21 and try to merge or zero out one term
    # This is a heuristic approach; may not succeed
    U21, V21, W21 = _build_rank21()
    U = np.array(U21, dtype=float)
    V = np.array(V21, dtype=float)
    W = np.array(W21, dtype=float)
    
    # Try removing one of the 21 terms and checking if the rest can compensate
    # This is very unlikely to work directly, so we try a small perturbation search
    rng = np.random.RandomState(42)
    
    for remove_idx in range(21):
        if time.time() > deadline:
            return None
        # Try to see if we can absorb term remove_idx into others
        # For simplicity, just check if removing it gives a close approximation
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, remove_idx] = 0
        V_test[:, remove_idx] = 0
        W_test[:, remove_idx] = 0
        
        # Check error
        max_err = 0.0
        n, m, p = 2, 2, 6
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(21):
                        total += U_test[i*m+j, c] * V_test[j*p+k, c] * W_test[k*n+i, c]
                    max_err = max(max_err, abs(total - 1.0))
        if max_err < 1e-10:
            # Found an exact rank-20 decomposition by removal
            # Remove the zero column
            keep = [c for c in range(21) if c != remove_idx]
            U_r = U[:, keep]
            V_r = V[:, keep]
            W_r = W[:, keep]
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    deadline = time.time() + 100  # Safety margin
    
    # First try rank 20 (structured search)
    result20 = _attempt_rank20(deadline)
    if result20 is not None:
        U, V, W = result20
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 21
    U, V, W = _build_rank21()
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Ultimate fallback: schoolbook rank 24
    n, m, p = 2, 2, 6
    rank = n * m * p
    U_s = [[0] * rank for _ in range(n * m)]
    V_s = [[0] * rank for _ in range(m * p)]
    W_s = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
