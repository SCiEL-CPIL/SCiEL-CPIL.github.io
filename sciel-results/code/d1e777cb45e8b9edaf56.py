"""Matrix multiplication 4x4 * 4x8 -> 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 via Strassen, rank 7.
    U: (4,7) rows indexed by i*2+j
    V: (4,7) rows indexed by j*2+k
    W: (4,7) rows indexed by k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1  # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1  # C00 += M5, C11 -= M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1  # C11 -= M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1  # C00 -= M7, C01 += M7
    
    return U, V, W


def _build_4x4_strassen():
    """Build 4x4 * 4x4 -> 4x4 via Kronecker of Strassen, rank 49.
    U: (16, 49) rows i*4+j
    V: (16, 49) rows j*4+k
    W: (16, 49) rows k*4+i
    """
    U2, V2, W2 = _strassen_2x2()
    r = 49
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u_val = U2[i1 * 2 + j1, c1]
                    if u_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val2 = U2[i2 * 2 + j2, c2]
                            if u_val2 == 0:
                                continue
                            u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[u_idx, c] += u_val * u_val2
            
            for j1 in range(2):
                for k1 in range(2):
                    v_val = V2[j1 * 2 + k1, c1]
                    if v_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_val2 = V2[j2 * 2 + k2, c2]
                            if v_val2 == 0:
                                continue
                            v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V[v_idx, c] += v_val * v_val2
            
            for k1 in range(2):
                for i1 in range(2):
                    w_val = W2[k1 * 2 + i1, c1]
                    if w_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_val2 = W2[k2 * 2 + i2, c2]
                            if w_val2 == 0:
                                continue
                            w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W[w_idx, c] += w_val * w_val2
    
    return U, V, W


def _build_decomposition():
    """Build 4x4 * 4x8 -> 4x8 decomposition using two 4x4*4x4 Strassen blocks.
    Rank: 98.
    U: (16, 98) rows i*4+j
    V: (32, 98) rows j*8+k
    W: (32, 98) rows k*4+i
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4_strassen()
    r = 98
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # First 4x4 * 4x4: B columns 0-3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # Second 4x4 * 4x4: B columns 4-7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, 49:98] = V4[j * 4 + (k - 4), 0:49]
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[(k - 4) * 4 + i, 0:49]
    
    return U, V, W


def _verify(U, V, W, n=4, m=4, p=8):
    """Verify tensor decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1"""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook(n=4, m=4, p=8):
    """Schoolbook decomposition, rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_8x8_embedding():
    """Try to use an 8x8x8 decomposition embedded into 4x4x8.
    Embed A(4x4) into 8x8 as [[A, 0], [0, 0]], B(4x8) into 8x8 as [[B, 0], [0, 0]].
    The product C(4x8) is the top-left 4x8 block of the 8x8 product.
    If we have a rank-r decomposition for 8x8x8, we can extract the relevant terms.
    However, most 8x8x8 decompositions have rank > 98, so this is unlikely to help.
    We'll just return None for now.
    """
    return None


def _build_decomposition_v2():
    """Alternative construction: use a 2x2 Strassen on the outer blocks.
    Split A(4x4) into 2x2 blocks of size 2x2, B(4x8) into 2x4 blocks.
    C(4x8) = [[C00, C01], [C10, C11]] where each Cij is 2x4.
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    We can use Strassen-like formulas with 2x2 * 2x4 products.
    Each 2x2 * 2x4 product has rank 16 (schoolbook) or can be done with 2x2 Strassen on each column pair?
    Actually, 2x2 * 2x4 is just 4 independent 2x2 * 2x1 products, each of rank 4, total 16.
    But we can share the A factors.
    
    Let's use the 4x4*4x4 approach which is already rank 98.
    """
    return _build_decomposition()


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s limit
    
    try:
        # First, try the standard rank-98 construction
        U, V, W = _build_decomposition()
        if time.time() > deadline:
            U, V, W = _schoolbook(n, m, p)
        elif _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Verification failed, use schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
