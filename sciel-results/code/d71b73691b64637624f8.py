"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Strategy: Tiled Strassen Decomposition
- Split 6x6 matrices into a 3x3 grid of 2x2 blocks.
- Use Strassen's algorithm (rank 7) for the 2x2 block multiplications.
- Total rank = 3 (I) * 3 (J) * 3 (K) * 7 (Strassen terms) = 189.
- This is an improvement over the naive rank of 216.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Evaluator kills at 150s; keep 10s margin


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive O(n*m*p) decomposition. Rank = n*m*p = 216 for 6x6x6."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_tiled_decomposition(n, m, p, block_size=2):
    """
    Builds a rank-189 decomposition using Strassen's algorithm on 2x2 blocks.
    n=m=p=6, block_size=2, blocks=3.
    """
    blocks = n // block_size  # 3
    # Strassen patterns for 2x2 matrix multiplication Z = X Y
    # Each pattern is (u_coeffs, v_coeffs, w_coeffs)
    # u, v, w are dicts mapping (row, col) to coefficient
    # Indices: 0,1 for rows/cols
    
    strassen_patterns = [
        # M0 = (X00+X11)(Y00+Y11) -> Z00 += M0, Z11 += M0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (X10+X11)Y00 -> Z10 += M1, Z11 -= M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = X00(Y01-Y11) -> Z01 += M2, Z11 += M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = X11(Y10-Y00) -> Z00 += M3, Z10 += M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (X00+X01)Y11 -> Z00 -= M4, Z01 += M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (X10-X00)(Y00+Y01) -> Z11 += M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (X01-X11)(Y10+Y11) -> Z00 += M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = (blocks ** 3) * len(strassen_patterns)  # 27 * 7 = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks):
        for J in range(blocks):
            for K in range(blocks):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U (A block I, J)
                    for (lr, lc), val in u_coeffs.items():
                        r = I * block_size + lr
                        c = J * block_size + lc
                        U[r * m + c][col_idx] += val
                    
                    # Update V (B block J, K)
                    for (lr, lc), val in v_coeffs.items():
                        r = J * block_size + lr
                        c = K * block_size + lc
                        V[r * p + c][col_idx] += val
                    
                    # Update W (C block I, K)
                    # W index convention: k*n + i, where k is row of C, i is col of C
                    for (lr, lc), val in w_coeffs.items():
                        k = I * block_size + lr
                        i = K * block_size + lc
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p, num_tests=3):
    """Verify that the tensor decomposition is exact using random tests."""
    U = np.array(decomp["U"], dtype=float)  # (n*m, rank)
    V = np.array(decomp["V"], dtype=float)  # (m*p, rank)
    W = np.array(decomp["W"], dtype=float)  # (p*n, rank)
    rank = U.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)  # W[k*n+i] indexed as (k,i)
            prod = Ur @ Vr
            C_computed += prod * Wr
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    if _remaining_time() < 5:
        # Emergency fallback
        return _build_naive_decomposition(n, m, p)
    
    # Try the Strassen-tiled decomposition (rank 189)
    decomp = _build_strassen_tiled_decomposition(n, m, p, block_size=2)
    
    if _remaining_time() < 5:
        return decomp
    
    # Verify the decomposition
    try:
        if _verify_decomposition(decomp, n, m, p, num_tests=3):
            return decomp
        else:
            # Fall back to naive
            return _build_naive_decomposition(n, m, p)
    except Exception:
        # Any error, fall back to naive
        return _build_naive_decomposition(n, m, p)


# Verify at import that our function works
if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    valid = _verify_decomposition(result, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
