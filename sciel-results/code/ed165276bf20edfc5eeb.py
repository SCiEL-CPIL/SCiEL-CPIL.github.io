import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def _build_strassen_block(U, V, W, base_r, k0, k1, p):
    """Add a Strassen 2x2x2 block for columns k0,k1 at rank offset base_r."""
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    r5 = base_r + 4
    r6 = base_r + 5
    r7 = base_r + 6

    # P1: (A00+A11)(B0k0+B1k1)
    U[0][r1] = 1
    U[3][r1] = 1
    V[0 * p + k0][r1] = 1
    V[1 * p + k1][r1] = 1

    # P2: (A10+A11)B0k0
    U[2][r2] = 1
    U[3][r2] = 1
    V[0 * p + k0][r2] = 1

    # P3: A00(B0k1-B1k1)
    U[0][r3] = 1
    V[0 * p + k1][r3] = 1
    V[1 * p + k1][r3] = -1

    # P4: A11(B1k0-B0k0)
    U[3][r4] = 1
    V[1 * p + k0][r4] = 1
    V[0 * p + k0][r4] = -1

    # P5: (A00+A01)B1k1
    U[0][r5] = 1
    U[1][r5] = 1
    V[1 * p + k1][r5] = 1

    # P6: (A10-A00)(B0k0+B0k1)
    U[2][r6] = 1
    U[0][r6] = -1
    V[0 * p + k0][r6] = 1
    V[0 * p + k1][r6] = 1

    # P7: (A01-A11)(B1k0+B1k1)
    U[1][r7] = 1
    U[3][r7] = -1
    V[1 * p + k0][r7] = 1
    V[1 * p + k1][r7] = 1

    # W coefficients for Strassen reconstruction
    W[2 * k0][r1] = 1
    W[2 * k0][r4] = 1
    W[2 * k0][r5] = -1
    W[2 * k0][r7] = 1

    W[2 * k0 + 1][r2] = 1
    W[2 * k0 + 1][r4] = 1

    W[2 * k1][r3] = 1
    W[2 * k1][r5] = 1

    W[2 * k1 + 1][r1] = 1
    W[2 * k1 + 1][r2] = -1
    W[2 * k1 + 1][r3] = 1
    W[2 * k1 + 1][r6] = 1


def _verify_decomposition(U, V, W, n=2, m=2, p=15, rank=None):
    """Verify that the tensor decomposition is exact using numpy for speed."""
    if rank is None:
        rank = len(U[0])
    U_np = np.array(U, dtype=np.int32)
    V_np = np.array(V, dtype=np.int32)
    W_np = np.array(W, dtype=np.int32)
    
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape for efficient computation
    # U: (n*m, rank)
    # V: (m*p, rank)  
    # W: (p*n, rank)
    
    # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                err = abs(float(s) - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _build_trivial_block(U, V, W, base_r, k, p, n=2, m=2):
    """Add a trivial 2x2x1 block for column k at rank offset base_r."""
    # C[0,k] = A00*B0k + A01*B1k
    # C[1,k] = A10*B0k + A11*B1k
    r = base_r
    U[0][r] = 1
    V[0*p+k][r] = 1
    W[k*n+0][r] = 1
    
    r += 1
    U[1][r] = 1
    V[1*p+k][r] = 1
    W[k*n+0][r] = 1
    
    r += 1
    U[2][r] = 1
    V[0*p+k][r] = 1
    W[k*n+1][r] = 1
    
    r += 1
    U[3][r] = 1
    V[1*p+k][r] = 1
    W[k*n+1][r] = 1


def construct():
    n, m, p = 2, 2, 15

    # Strategy: Use Strassen blocks for pairs of columns, plus handle remaining column.
    # 15 columns = 7 pairs (14 cols) + 1 remaining col
    # Each Strassen block: rank 7
    # Trivial single column: rank 4
    # Total: 7*7 + 4 = 53

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 30 x rank

    # Build 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        base_r = 7 * block
        _build_strassen_block(U, V, W, base_r, k0, k1, p)

    # Handle column 14 trivially (rank 4)
    base_r = 49
    _build_trivial_block(U, V, W, base_r, 14, p, n, m)

    # Verify
    if time.time() < _DEADLINE - 5:
        max_err = _verify_decomposition(U, V, W, n, m, p, rank)
        if max_err > 1e-10:
            # Fallback: return a known-good trivial decomposition (rank 60)
            rank = 60
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                        r += 1

    return {"U": U, "V": V, "W": W}
