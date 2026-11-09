import time
import numpy as np

DIMENSIONS = [4, 4, 5]

DEADLINE_FRAC = 0.90          # return before ~150s wall-clock
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

# ----------------------------------------------------------------------
# Exact schoolbook decomposition (rank n*m*p = 80). Guaranteed correct.
# ----------------------------------------------------------------------
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


# ----------------------------------------------------------------------
# Verify a candidate (U,V,W) exactly reproduces matmul C = A @ B.
# C[i,k] = sum_r U[i,r]*V[k,r]*W[k*n+i,r]   (official trace convention)
# ----------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    # Use a generic matrix to avoid degenerate cases
    A = np.random.default_rng(42).random((n, m))
    B = np.random.default_rng(123).random((m, p))
    Ctrue = A @ B
    # build C from tensor
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)


# ----------------------------------------------------------------------
# 2x2x2 Strassen block (rank 7) used at the lowest recursion level.
# ----------------------------------------------------------------------
def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    """
    Returns U, V, W for 2x2x2 tensor using Strassen (Rank 7).
    Assumes nrows=2, ncolsm=2, ncolsk=2.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Standard Strassen
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> flat: 0, 1, 2, 3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W


# ----------------------------------------------------------------------
# Recursively build the 4x4x4 Strassen block via block decomposition,
# then append 4 extra columns (one per extra k) with schoolbook on A-rows.
# ----------------------------------------------------------------------
def _strassen_444_and_append(n, m, p):
    # A is n x m = 4 x 4, B is m x p = 4 x 5.
    # Split A into 2x2 blocks of 2x2: A[r,c] = A[2r:2r+2, 2c:2c+2]
    # Split B into 2x2(over k) blocks but k has 5 -> we handle k=0..3 via 2x2 blocks
    # and k=4 separately with schoolbook.
    U4, V4, W4 = _make_4x4x4_strassen()
    # Append extra columns for k=4: for each i (0..3) and j (0..3), one column.
    R = U4.shape[1]
    extra = n * m   # 16
    U = np.zeros((n * m, R + extra), dtype=float)
    V = np.zeros((m * p, R + extra), dtype=float)
    W = np.zeros((p * n, R + extra), dtype=float)
    U[:, :R] = U4
    V[:, :R] = V4
    W[:, :R] = W4
    k = 4
    c = R
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    return U, V, W


def _make_4x4x4_strassen():
    """
    Builds 4x4x4 Strassen decomposition (Rank 49).
    Uses 2x2 blocks of 2x2.
    """
    n = m = p = 4
    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)
    
    inner_U, inner_V, inner_W = _make_2x2x2_strassen(2, 2, 2)
    
    # Top level Strassen combinations
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Define top-level coefficients for A and B for each of the 7 products.
    # Format: list of (block_index, coeff)
    # Block indices: 0=00, 1=01, 2=10, 3=11
    top_A = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, 1)],   # M2
        [(0, 1)],           # M3
        [(3, 1)],           # M4
        [(0, 1), (1, 1)],   # M5
        [(2, 1), (0, -1)],  # M6
        [(1, 1), (3, -1)],  # M7
    ]
    top_B = [
        [(0, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(2, 1), (3, -1)],  # M3
        [(1, 1), (0, -1)],  # M4
        [(3, 1)],           # M5
        [(0, 1), (1, 1)],   # M6
        [(2, 1), (3, 1)],   # M7
    ]
    
    # C combination for top level
    # C00 = M1 + M4 + M5 + M7
    # C01 = M5 + M6
    # C10 = M3 + M5
    # C11 = M2 + M3 + M4
    top_C = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3]
    ]
    
    for t in range(7):
        col_base = t * 7
        # For each inner multiplication tp (0..6)
        for tp in range(7):
            col = col_base + tp
            
            # Accumulate U
            for b_idx, coef in top_A[t]:
                br, bc = b_idx // 2, b_idx % 2
                for ii in range(2):
                    for jj in range(2):
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        U[gi * m + gj, col] += coef * inner_U[ii * 2 + jj, tp]
            
            # Accumulate V
            for b_idx, coef in top_B[t]:
                br, bc = b_idx // 2, b_idx % 2
                for jj in range(2):
                    for kk in range(2):
                        gj = br * 2 + jj
                        gk = bc * 2 + kk
                        V[gj * p + gk, col] += coef * inner_V[jj * 2 + kk, tp]
            
            # Accumulate W
            for c_idx in range(4):
                if t in top_C[c_idx]:
                    cr, cc = c_idx // 2, c_idx % 2
                    for ii in range(2):
                        for kk in range(2):
                            gi = cr * 2 + ii
                            gk = cc * 2 + kk
                            W[gk * n + gi, col] += inner_W[kk * 2 + ii, tp]
                            
    return U, V, W


# ----------------------------------------------------------------------
# Final: build candidates, verify, pick minimal verified rank; fallback schoolbook.
# ----------------------------------------------------------------------
def construct():
    n, m, p = DIMENSIONS
    best = None
    # Candidate 1: schoolbook (always correct)
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    # Candidate 2: 4x4x4 Strassen (49) + 16 extra = 65
    if not _deadline():
        try:
            U2, V2, W2 = _strassen_444_and_append(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    # Candidate 3: plain schoolbook if Strassen failed (already best)
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
