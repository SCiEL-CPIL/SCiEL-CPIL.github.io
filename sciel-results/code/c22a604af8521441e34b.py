"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms: should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms: should equal 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                    if abs(val) > 1e-9:
                        return False
    return True

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
    return U, V, W

def _strassen_block(n, m, p):
    """
    Construct rank-56 decomposition for 2x2x16 matrix multiplication.
    Uses 8 independent Strassen (rank 7) decompositions for the 8 2x2 blocks.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # Indices for this block t
        # B indices: 2t, 2t+1 (row 0), 16+2t, 16+2t+1 (row 1)
        # C indices: 4t, 4t+1 (col 2t, 2t+1 row 0), 4t+2, 4t+3 (col 2t, 2t+1 row 1)
        # A indices: 0, 1, 2, 3 (standard 2x2)
        
        # Strassen intermediates for 2x2x2
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # U coeffs (A): A00=0, A01=1, A10=2, A11=3
        # V coeffs (B): B00=2t, B01=2t+1, B10=16+2t, B11=16+2t+1
        # W coeffs (C): C00=4t, C10=4t+1, C01=4t+2, C11=4t+3
        
        intermediates = [
            # M1
            {'U': [(0, 1), (3, 1)], 'V': [(2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1), (4*t+3, 1)]},
            # M2
            {'U': [(2, 1), (3, 1)], 'V': [(2*t, 1)], 'W': [(4*t+1, 1), (4*t+3, -1)]},
            # M3
            {'U': [(0, 1)], 'V': [(2*t+1, 1), (16+2*t+1, -1)], 'W': [(4*t+2, 1), (4*t+3, 1)]},
            # M4
            {'U': [(3, 1)], 'V': [(16+2*t, 1), (2*t, -1)], 'W': [(4*t, 1), (4*t+1, 1)]},
            # M5
            {'U': [(0, 1), (1, 1)], 'V': [(16+2*t+1, 1)], 'W': [(4*t, -1), (4*t+2, 1)]},
            # M6
            {'U': [(2, 1), (0, -1)], 'V': [(2*t, 1), (2*t+1, 1)], 'W': [(4*t+3, 1)]},
            # M7
            {'U': [(1, 1), (3, -1)], 'V': [(16+2*t, 1), (16+2*t+1, 1)], 'W': [(4*t, 1)]}
        ]
        
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strassen-based construction (Rank 56)
    U, V, W = _strassen_block(n, m, p)
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (Rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
