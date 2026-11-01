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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that no spurious terms exist
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
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
    For multiplying (n x m) by (m x p) where n=m=2, p=16.
    The 2x16 matrix can be seen as 8 independent 2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    
    For block t (0..7), the columns in B are at indices 2t and 2t+1.
    Output rows in C are at indices 2t*n+0, 2t*n+1, (2t+1)*n+0, (2t+1)*n+1.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # For this block, B entries are at V indices:
        # B[0, 2t] -> V[0 * p + 2t] = V[2t]
        # B[0, 2t+1] -> V[0 * p + 2t+1] = V[2t+1]
        # B[1, 2t] -> V[1 * p + 2t] = V[p + 2t] = V[16 + 2t]
        # B[1, 2t+1] -> V[1 * p + 2t+1] = V[16 + 2t + 1]
        
        # W entries:
        # C[0, 2t] -> W[2t * n + 0] = W[4t]
        # C[1, 2t] -> W[2t * n + 1] = W[4t+1]
        # C[0, 2t+1] -> W[(2t+1) * n + 0] = W[4t+2]
        # C[1, 2t+1] -> W[(2t+1) * n + 1] = W[4t+3]
        
        # Strassen intermediates:
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
        
        # U coefficients (A indices: A00=0, A01=1, A10=2, A11=3)
        # M1: A00=1, A11=1
        # M2: A10=1, A11=1
        # M3: A00=1
        # M4: A11=1
        # M5: A00=1, A01=1
        # M6: A10=1, A00=-1
        # M7: A01=1, A11=-1
        
        # V coefficients
        # M1: B00=1, B11=1
        # M2: B00=1
        # M3: B01=1, B11=-1
        # M4: B10=1, B00=-1
        # M5: B11=1
        # M6: B00=1, B01=1
        # M7: B10=1, B11=1
        
        # W coefficients
        # M1: C00=1, C11=1
        # M2: C10=1, C11=-1
        # M3: C01=1, C11=1
        # M4: C00=1, C10=1
        # M5: C00=-1, C01=1
        # M6: C11=1
        # M7: C00=1
        
        # Let's define the entries for each intermediate M
        # U entries: (row, coeff)
        # V entries: (col, coeff)
        # W entries: (row, coeff)
        
        intermediates = [
            # M1
            {
                'U': [(0, 1), (3, 1)],
                'V': [(2*t, 1), (16+2*t+1, 1)],
                'W': [(4*t, 1), (4*t+3, 1)]
            },
            # M2
            {
                'U': [(2, 1), (3, 1)],
                'V': [(2*t, 1)],
                'W': [(4*t+1, 1), (4*t+3, -1)]
            },
            # M3
            {
                'U': [(0, 1)],
                'V': [(2*t+1, 1), (16+2*t+1, -1)],
                'W': [(4*t+2, 1), (4*t+3, 1)]
            },
            # M4
            {
                'U': [(3, 1)],
                'V': [(16+2*t, 1), (2*t, -1)],
                'W': [(4*t, 1), (4*t+1, 1)]
            },
            # M5
            {
                'U': [(0, 1), (1, 1)],
                'V': [(16+2*t+1, 1)],
                'W': [(4*t, -1), (4*t+2, 1)]
            },
            # M6
            {
                'U': [(2, 1), (0, -1)],
                'V': [(2*t, 1), (2*t+1, 1)],
                'W': [(4*t+3, 1)]
            },
            # M7
            {
                'U': [(1, 1), (3, -1)],
                'V': [(16+2*t, 1), (16+2*t+1, 1)],
                'W': [(4*t, 1)]
            }
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
    deadline = start_time + 120
    
    # Try Strassen-based construction (rank 56)
    U, V, W = _strassen_block(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
