# EVOLVE-BLOCK-START
import time
import math

def _build_strassen():
    """Return the verified rank-7 Strassen decomposition."""
    n, m, p = DIMENSIONS
    r = 7
    U = [[0.0] * r for _ in range(4)]
    V = [[0.0] * r for _ in range(4)]
    W = [[0.0] * r for _ in range(4)]

    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0

    # M2 = (A10+A11)*B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0

    # M3 = A00*(B01-B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0

    # M4 = A11*(B10-B00)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0

    # M5 = (A00+A01)*B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0

    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0

    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    Returns True if all entries match, False otherwise.
    """
    rank = len(U[0])
    # Check all 8 tensor entries (i, k, j) for n=m=p=2.
    for i in range(n):
        for k in range(m):
            for j in range(p):
                idx_u = i * m + k
                idx_v = k * p + j
                idx_w = j * n + i
                val = 0.0
                for s in range(rank):
                    val += U[idx_u][s] * V[idx_v][s] * W[idx_w][s]
                if abs(val - 1.0) > 1e-9:
                    return False
                
    # Check invalid entries
    valid_set = set()
    for i in range(n):
        for k in range(m):
            for j in range(p):
                valid_set.add((i*m+k, k*p+j, j*n+i))
    
    for ru in range(n*m):
        for rv in range(m*p):
            for rw in range(p*n):
                if (ru, rv, rw) not in valid_set:
                    val = 0.0
                    for s in range(rank):
                        val += U[ru][s] * V[rv][s] * W[rw][s]
                    if abs(val) > 1e-9:
                        return False
    return True

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # 1. Construct the known optimal rank-7 Strassen decomposition
    res7 = _build_strassen()
    
    # 2. Verify Strassen (should always pass)
    if _verify_decomposition(res7['U'], res7['V'], res7['W'], 2, 2, 2):
        return res7
    else:
        # This should never happen
        raise RuntimeError("Strassen decomposition verification failed")

# EVOLVE-BLOCK-END
