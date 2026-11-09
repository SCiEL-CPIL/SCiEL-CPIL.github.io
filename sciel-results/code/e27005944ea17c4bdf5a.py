"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _is_half_int(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half_int(x):
    return round(x * 2) / 2.0

def _schoolbook(n, m, p):
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

def _strassen_decomposition():
    """
    Construct a rank-33 decomposition for 2x3 * 3x6 matrix multiplication.
    Uses Strassen for the 2x2 * 2x6 part (3 blocks of 2x2) and schoolbook for the 2x1 * 1x6 part.
    """
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 3, 4
    # B indices: 00, 01, 10, 11 -> 0, 1, 6, 7 (for block 0)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (for block 0, k=0,1)
    
    strassen_A = [
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1]   # M7: A12 - A22
    ]
    
    strassen_B = [
        [1, 1, 1, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [0, 0, 1, -1],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1]    # M7: B21 + B22
    ]
    
    strassen_C = [
        [1, 0, 0, 1],   # C11: M1 + M4 - M5 + M7
        [0, 1, 0, 0],   # C12: M3 + M5
        [0, 0, 1, 0],   # C21: M2 + M4
        [1, 0, 0, 1]    # C22: M1 - M2 + M3 + M6 -> Wait, C22 = M1 - M2 + M3 + M6
    ]
    # Correct C22: M1 - M2 + M3 + M6
    # M1: 1, M2: -1, M3: 1, M6: 1
    # So coeffs for M1..M7:
    # C11: 1, 0, 0, 1, -1, 0, 1
    # C12: 0, 0, 1, 0, 1, 0, 0
    # C21: 0, 1, 0, 1, 0, 0, 0
    # C22: 1, -1, 1, 0, 0, 1, 0
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]
    
    # Fill U, V, W for the 3 Strassen blocks
    for t in range(3):
        for s in range(7):
            c = t * 7 + s
            
            # U: A indices 0, 1, 3, 4
            U[0, c] = strassen_A[s][0]
            U[1, c] = strassen_A[s][1]
            U[3, c] = strassen_A[s][2]
            U[4, c] = strassen_A[s][3]
            
            # V: B indices for block t
            # B rows 0, 1. Cols 2t, 2t+1.
            # Indices: 0*6+2t, 0*6+2t+1, 1*6+2t, 1*6+2t+1
            v_idx = [2*t, 2*t+1, 6+2*t, 7+2*t]
            V[v_idx[0], c] = strassen_B[s][0]
            V[v_idx[1], c] = strassen_B[s][1]
            V[v_idx[2], c] = strassen_B[s][2]
            V[v_idx[3], c] = strassen_B[s][3]
            
            # W: C indices for block t
            # C rows 0, 1. Cols 2t, 2t+1.
            # Indices: 2t*2+0, 2t*2+1, (2t+1)*2+0, (2t+1)*2+1
            w_idx = [4*t, 4*t+1, 4*t+2, 4*t+3]
            # C_coeffs[s] gives coeffs for M1..M7 for each C element?
            # No, C_coeffs is defined per C element.
            # We need to sum over s?
            # No, each term c corresponds to one product M_s.
            # The contribution of M_s to C_ij is C_coeffs[i, j, s].
            # So W[k, i, c] should be C_coeffs[i, j, s] where k=j.
            # Wait, W is indexed by k (col of C) and i (row of C).
            # So for a given c (which is M_s for block t),
            # W[k, i, c] = C_coeffs[i, k, s] if k is in block t.
            
            for i in range(2):
                for k in range(2):
                    # k is local col index 0 or 1
                    global_k = 2*t + k
                    w_idx_global = global_k * 2 + i
                    W[w_idx_global, c] = C_coeffs[i, k, s]
    
    # Fill U, V, W for the schoolbook part (j=2)
    # A index 2, 5. B index 12..17.
    for i in range(2):
        for k in range(6):
            c = 21 + i * 6 + k
            U[i*3 + 2, c] = 1
            V[2*6 + k, c] = 1
            W[k*2 + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Try the Strassen-based decomposition
    try:
        res = _strassen_decomposition()
        U, V, W = res["U"], res["V"], res["W"]
        if _verify(U, V, W, n, m, p, 33):
            return res
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
