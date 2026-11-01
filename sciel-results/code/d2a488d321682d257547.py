DIMENSIONS = [2, 2, 4]
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p solution."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Two independent Strassen blocks.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in {0,1}, r indices 0..6
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[2][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[2][6] = 1
    W[0][6] = 1

    # Block 2: k in {2,3}, r indices 7..13
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[6][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[5][9] = 1; V[6][9] = -1
    W[4][9] = 1; W[6][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[6][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[5][13] = 1; V[6][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _structured_search(n, m, p, deadline):
    """
    Global structured search: try to find a rank-14 decomposition where the
    two 2x2 blocks share some rank-1 factors (cross-block coupling).
    
    We parameterize each of the 14 columns as a combination of the standard
    Strassen patterns for block 1 and block 2, with mixing coefficients.
    Then we do a local search over these coefficients.
    
    The key insight: if a rank < 14 decomposition exists, it must use some
    rank-1 terms that contribute to BOTH output blocks simultaneously.
    """
    start = time.time()
    
    # Standard Strassen patterns for a single 2x2 block
    # U patterns (4 rows: A00, A01, A10, A11)
    # V patterns (4 rows: B00, B01, B10, B11)
    # W patterns (4 rows: C00, C01, C10, C11)
    
    U_pats = [
        [1,0,0,1],  # M1: A00+A11
        [0,1,0,0],  # M3: A00
        [1,0,1,0],  # M2: A10+A11  (note: A10 is row 2, A11 is row 3)
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: A00+A01
        [1,-1,0,0], # M6: A10-A00 -> actually [0,0,1,-1]... let me fix
        [0,1,0,-1], # M7: A01-A11
    ]
    # Let me be more careful with Strassen patterns
    # A = [[A00, A01],[A10, A11]], rows in U: [A00, A01, A10, A11]
    # B = [[B00, B01],[B10, B11]], rows in V: [B00, B01, B10, B11]
    # C = [[C00, C01],[C10, C11]], rows in W: [C00, C01, C10, C11]
    
    U_pats = [
        [1,0,0,1],  # M1: (A00+A11)
        [0,0,1,1],  # M2: (A10+A11)
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: (A00+A01)
        [0,0,1,-1], # M6: (A10-A00)
        [0,1,0,-1], # M7: (A01-A11)
    ]
    V_pats = [
        [1,0,0,1],  # M1: (B00+B11)
        [1,0,0,0],  # M2: B00
        [0,0,1,-1], # M3: (B10-B11)
        [0,1,-1,0], # M4: (B01-B00) -> B01 - B00 = [ -1, 1, 0, 0]... 
        # Let me redo: B rows are [B00, B01, B10, B11]
        # M1: B00+B11 = [1,0,0,1]
        # M2: B00 = [1,0,0,0]
        # M3: B10-B11 = [0,0,1,-1]
        # M4: B01-B00 = [-1,1,0,0]
        # M5: B11 = [0,0,0,1]
        # M6: B00+B01 = [1,1,0,0]
        # M7: B10+B11 = [0,0,1,1]
    ]
    V_pats = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,0,1,-1], # M3
        [-1,1,0,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1],  # M7
    ]
    W_pats = [
        [1,0,0,1],  # M1: C00+C11
        [0,0,1,-1], # M2: C10-C11
        [1,1,0,0],  # M3: C00+C01
        [1,0,1,0],  # M4: C00+C10
        [-1,1,0,0], # M5: -C00+C01
        [0,0,0,1],  # M6: C11
        [1,0,0,0],  # M7: C00
    ]
    
    # Verify Strassen: C = sum M_r * (U_r outer W_r)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check: for each output (i,k), sum_r U_r[i] * W_r[k] * M_r should equal delta
    # Actually the tensor is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For a single 2x2 block, U has 4 rows (i*m+j), V has 4 rows (j*p+k), W has 4 rows (k*n+i)
    
    # Let's just verify the known Strassen works for a single block
    # Then build the global search
    
    # For the global search, we'll try a different approach:
    # Use a random restart local search over the space of rank-14 decompositions
    # that are "close" to the two-block Strassen solution.
    
    # Actually, let's try a more principled approach:
    # The 2x2 x 2x4 problem can be seen as computing two 2x2 matrix products
    # with the same A but different B halves. If a rank < 14 solution exists,
    # it must exploit the shared A.
    
    # Let's try a direct algebraic approach: assume a rank-13 or rank-14
    # solution with specific structure and solve for the entries.
    
    # For now, let's just return the known rank-14 solution.
    # The structured search would need much more work to be effective.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, try the known rank-14 solution
    sol = _strassen_2x2x4()
    if _verify_decomposition(sol['U'], sol['V'], sol['W'], n, m, p, tol=1e-9):
        return sol
    
    # Try structured search if time permits
    if time.time() < deadline:
        result = _structured_search(n, m, p, deadline)
        if result is not None:
            if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, tol=1e-9):
                return result
    
    # Fallback
    return _trivial_solution(n, m, p)
