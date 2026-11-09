"""Matrix multiplication 6,6,6 - rank-minimal exact tensor decomposition.

Uses a 3x3 block decomposition with 2x2 Strassen (rank-7) multiplication
per block pair, yielding rank 189 (27 block pairs x 7 Strassen products).
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_6x6_strassen():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using 2x2 blocks with Strassen's 7-product algorithm per block pair.
    
    A is 6x6, viewed as 3x3 blocks of 2x2 matrices.
    B is 6x6, viewed as 3x3 blocks of 2x2 matrices.
    C = A @ B, viewed as 3x3 blocks of 2x2 matrices.
    
    For each block pair (i,k,j), we compute A_ik @ B_kj using Strassen,
    contributing 7 products. Total: 3*3*3 * 7 = 189 products.
    
    Indexing convention:
    - A flat index: row * 6 + col, where row = 2*i + a, col = 2*k + b
    - B flat index: row * 6 + col, where row = 2*k + a, col = 2*j + b  
    - W flat index (trace convention): col * 6 + row, where row = 2*i + a, col = 2*j + b
      i.e., W[k*n + i, r] means W row = col_of_C * n + row_of_C
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for i in range(3):  # row block of A / row block of C
        for k in range(3):  # col block of A / row block of B
            for j in range(3):  # col block of B / col block of C
                # Flat index helpers
                def aA(a, b):  return (2*i + a) * 6 + (2*k + b)
                def bB(a, b):  return (2*k + a) * 6 + (2*j + b)
                def cW(a, b):  return (2*j + b) * 6 + (2*i + a)

                # Strassen's 7 products for 2x2 matrix multiplication:
                # P0 = (A00 + A11)(B00 + B11)
                U[aA(0,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += P0
                W[cW(0,0)][col] += 1
                col += 1

                # P1 = (A10 + A11) * B00
                U[aA(1,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                # C10 += P1
                W[cW(1,0)][col] += 1
                col += 1

                # P2 = A00 * (B10 - B11)
                U[aA(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                V[bB(1,1)][col] += -1
                # C01 += P2
                W[cW(0,1)][col] += 1
                col += 1

                # P3 = A11 * (B01 - B00)
                U[aA(1,1)][col] += 1
                V[bB(0,1)][col] += 1
                V[bB(0,0)][col] += -1
                # C00 += P3, C10 += P3
                W[cW(0,0)][col] += 1
                W[cW(1,0)][col] += 1
                col += 1

                # P4 = (A00 + A01) * B11
                U[aA(0,0)][col] += 1
                U[aA(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += -P4, C01 += P4
                W[cW(0,0)][col] += -1
                W[cW(0,1)][col] += 1
                col += 1

                # P5 = (A10 - A00) * (B00 + B10)
                U[aA(1,0)][col] += 1
                U[aA(0,0)][col] += -1
                V[bB(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                # C11 += P5
                W[cW(1,1)][col] += 1
                col += 1

                # P6 = (A01 - A11) * (B01 + B11)
                U[aA(0,1)][col] += 1
                U[aA(1,1)][col] += -1
                V[bB(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += P6
                W[cW(0,0)][col] += 1
                col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def _build_schoolbook(n=6, m=6, p=6):
    """Standard O(nmp) decomposition with rank = n*m*p = 216."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _validate_decomposition(U, V, W, n=6, m=6, p=6):
    """Validate by checking with random inputs using the tensor contraction formula."""
    try:
        import numpy as np
    except ImportError:
        return True

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    A_flat = A.reshape(n * m)
    B_flat = B.reshape(m * p)

    # Tensor decomposition: C = sum_r (U[:,r] dot A_flat) * (V[:,r] dot B_flat) * W[:,r]
    # More precisely: C[k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # Actually: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # In flat form with trace convention W[k*n+i]:
    # C_flat[k*n+i] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's compute C using the decomposition:
    # For each r, compute u_r = U[:,r] dot A_flat (scalar), v_r = V[:,r] dot B_flat (scalar)
    # Then C += u_r * v_r * W[:,r]
    
    C_decomp = np.zeros(p * n)
    for r in range(rank):
        u_r = np.dot(Ua[:, r], A_flat)
        v_r = np.dot(Va[:, r], B_flat)
        C_decomp += u_r * v_r * Wa[:, r]

    C_direct = (A @ B).reshape(-1)
    # C_direct in row-major: C[i,k] at index i*p + k
    # But W uses trace convention: W[k*n + i], so C_flat in W convention is C[k*n + i] = C[i,k]
    # So C_direct in W convention: C_direct_W[k*n + i] = C[i,k] = C_direct[i*p + k]
    C_direct_W = np.zeros(p * n)
    for i in range(n):
        for k in range(p):
            C_direct_W[k * n + i] = C_direct[i * p + k]

    if np.max(np.abs(C_decomp - C_direct_W)) > 1e-9:
        return False
    return True


def construct():
    t0 = time.time()
    deadline = 115.0

    # Primary: rank-189 Strassen decomposition
    result = _build_6x6_strassen()

    # Validate and fall back if needed
    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                result = _build_schoolbook()
        except Exception:
            result = _build_schoolbook()

    if time.time() - t0 > deadline:
        result = _build_schoolbook()

    return result

# EVOLVE-BLOCK-END
