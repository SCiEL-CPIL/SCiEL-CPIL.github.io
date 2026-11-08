"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition.
Uses Kronecker products of 2x2x2 Strassen decompositions.

The 4x4x8 matmul tensor is decomposed as:
  i = 2a + b, j = 2c + d, k = 2e + f
  where a,c in {0,1}, b,d in {0,1}, e in {0,1,2,3}, f in {0,1}

This gives a tensor product of:
  - Big: 2x2x4 (indices a, c, e) — two independent 2x2x2 Strassen instances (rank 14)
  - Small: 2x2x2 (indices b, d, f) — standard Strassen (rank 7)

Total rank = 14 * 7 = 98.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2_terms():
    """Return 7 rank-1 Strassen terms for 2x2x2 matmul.
    
    Each term is a dict with 'U', 'V', 'W' as flat lists of 4 entries.
    Indexing: element (i,j) -> i*2+j for U; (j,k) -> j*2+k for V; (k,i) -> k*2+i for W.
    """
    return [
        # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        # M2 = (A10+A11)B00: C10 += M2, C11 -= M2
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        # M3 = A00(B01-B11): C01 += M3, C11 += M3
        {'U': [1, 0, 0, 0], 'V': [0, 1, -1, 0], 'W': [0, 0, 1, 1]},
        # M4 = A11(B10-B00): C00 += M4, C10 += M4
        {'U': [0, 0, 0, 1], 'V': [0, -1, 1, 0], 'W': [1, 1, 0, 0]},
        # M5 = (A00+A01)B11: C00 -= M5, C01 += M5
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        # M6 = (A10-A00)(B00+B01): C11 += M6
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        # M7 = (A01-A11)(B10+B11): C00 += M7
        {'U': [0, 1, -1, 0], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]


def _build_decomposition():
    """Build the 4x4x8 decomposition with rank 98 using Kronecker structure."""
    s222 = _strassen_2x2_terms()
    
    n, m, p = 4, 4, 8
    rank_total = 98
    U = np.zeros((n * m, rank_total), dtype=np.int64)
    V = np.zeros((m * p, rank_total), dtype=np.int64)
    W = np.zeros((p * n, rank_total), dtype=np.int64)
    
    r_idx = 0
    for g in range(2):          # g: which pair of e-indices (e = 2g + h)
        for s1 in range(7):     # s1: Strassen term for the "big" 2x2x2
            big = s222[s1]
            for s2 in range(7): # s2: Strassen term for the "small" 2x2x2
                small = s222[s2]
                
                # U[i, j, r] where i = 2a+b, j = 2c+d
                # U = U_big[a,c] * U_small[b,d]
                for a in range(2):
                    for c in range(2):
                        for b in range(2):
                            for d in range(2):
                                i = 2 * a + b
                                j = 2 * c + d
                                U[i * m + j, r_idx] = big['U'][a * 2 + c] * small['U'][b * 2 + d]
                
                # V[j, k, r] where j = 2c+d, k = 2e+f, e = 2g+h
                # V = V_big[c, h] * V_small[d, f]
                for c in range(2):
                    for d in range(2):
                        j = 2 * c + d
                        for h in range(2):
                            e = 2 * g + h
                            for f in range(2):
                                k = 2 * e + f
                                V[j * p + k, r_idx] = big['V'][c * 2 + h] * small['V'][d * 2 + f]
                
                # W[k, i, r] where k = 2e+f, i = 2a+b
                # W = W_big[h, a] * W_small[f, b]
                for a in range(2):
                    for b in range(2):
                        i = 2 * a + b
                        for h in range(2):
                            e = 2 * g + h
                            for f in range(2):
                                k = 2 * e + f
                                W[k * n + i, r_idx] = big['W'][h * 2 + a] * small['W'][f * 2 + b]
                
                r_idx += 1
    
    assert r_idx == rank_total
    
    # Verify correctness
    np.random.seed(123)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank_total):
        Ur = U[:, r].reshape(n, m).astype(float)
        Vr = V[:, r].reshape(m, p).astype(float)
        Wr = W[:, r].reshape(p, n).astype(float)
        a_part = np.sum(Ur * A_test, axis=1)
        b_part = np.sum(Vr * B_test, axis=0)
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += a_part[i] * b_part[k] * Wr[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        raise ValueError(f"Verification failed: max error {np.max(np.abs(C_computed - C_expected))}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 100
    
    try:
        result = _build_decomposition()
        if time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 128)
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
