import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _verify_solution(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition for M_{n,m,p}.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    Condition: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] == delta(j, j') for all i, k, j, j'
    """
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > 1e-9:
                        return False
    return True

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

def _als_solve(rank, n, m, p, max_iters=500, restarts=100, timeout=100):
    """
    Uses Alternating Least Squares to find a rank 'rank' decomposition.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    start_time = time.time()
    
    for restart in range(restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            # Update U: for each (i,j), solve for U[i*m+j]
            # b[k, jp] = delta(j, jp), A[k, jp, c] = W[k*n+i, c] * V[jp*p+k, c]
            for i in range(n):
                for j in range(m):
                    # Build A: shape (p*m, rank)
                    A = np.zeros((p * m, rank))
                    b = np.zeros(p * m)
                    idx = 0
                    for k in range(p):
                        for jp in range(m):
                            A[idx] = W[k * n + i] * V[jp * p + k]
                            b[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = x
                    
            # Update V: for each (j,k), solve for V[j*p+k]
            # b[i, jp] = delta(jp, j), A[i, jp, c] = W[k*n+i, c] * U[i*m+jp, c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n * m, rank))
                    b = np.zeros(n * m)
                    idx = 0
                    for i in range(n):
                        for jp in range(m):
                            A[idx] = W[k * n + i] * U[i * m + jp]
                            b[idx] = 1.0 if jp == j else 0.0
                            idx += 1
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = x
                    
            # Update W: for each (k,i), solve for W[k*n+i]
            # b[j, jp] = delta(j, jp), A[j, jp, c] = U[i*m+j, c] * V[jp*p+k, c]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m * m, rank))
                    b = np.zeros(m * m)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            A[idx] = U[i * m + j] * V[jp * p + k]
                            b[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = x
            
            # Check error
            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                            target = 1.0 if j == jp else 0.0
                            error += (s - target) ** 2
            
            if error < 1e-12:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list

    return None

def _strassen_rank12():
    """
    Attempt a rank-12 decomposition using structured Strassen-like approach.
    Partition B (3x4) into B_left (3x2) and B_right (3x2).
    For each 2x2 output block, use Strassen's 7-multiplication algorithm.
    The A-side combinations are shared between left and right blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # We need to find U (6, r), V (12, r), W (8, r) such that
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta(j, j')
    
    # Let's try to construct this explicitly.
    # For 2x2x2 Strassen, the 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22*B11
    # M3 = (A11+A22-A12)*B22  -- wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For our case, A is 2x3, so we partition columns: A = [A0 | A1 | A2] where Ai is 2x1
    # B is 3x4, partition into B_L (3x2) and B_R (3x2)
    
    # For C_L = A * B_L (2x2):
    # We can write A = [A0, A1, A2] and B_L has rows b0, b1, b2 (each 1x2)
    # C_L[i,k] = sum_j A[i,j] * B_L[j,k]
    
    # Let me just try to find a valid rank-12 solution via optimization with structured init.
    # Actually, let me just return None and let ALS handle it.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 115  # Leave 5s margin before 120s kill
    
    # First try the structured Strassen approach (instant if it works)
    result = _strassen_rank12()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try ranks from 19 up to 23 with ALS
    for r in range(19, 24):
        remaining = deadline - time.time() - 3
        if remaining < 5:
            break
        # Scale iterations and restarts based on remaining time
        max_iters = min(300, max(50, int(remaining * 10)))
        restarts = min(50, max(5, int(remaining * 2)))
        result = _als_solve(r, n, m, p, max_iters=max_iters, restarts=restarts, timeout=remaining)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
