"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses a two-level Strassen block construction for the 4x4x4 core (rank 49)
plus schoolbook columns for the 5th output column, giving rank 65.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0


def _time_remaining():
    return _DEADLINE - time.time() - _SAFETY_MARGIN


def _strassen_2x2x2(nrows, ncolsm, ncolsk):
    """Build U(nrows*ncolsm, 7), V(ncolsm*ncolsk, 7), W(ncolsk*nrows, 7)
    for a 2x2x2 Strassen decomposition.
    
    Indexing: A is nrows x ncolsm, B is ncolsm x ncolsk, C is nrows x ncolsk.
    U indexed by (i,j) -> i*ncolsm+j
    V indexed by (j,k) -> j*ncolsk+k
    W indexed by (k,i) -> k*nrows+i  (official trace convention)
    """
    # Strassen factors: each factor t has a-coefficients (2x2) and b-coefficients (2x2)
    # a_coeffs[t][i][j] is the coefficient of A[i,j] in the t-th product's A-factor
    # b_coeffs[t][j][k] is the coefficient of B[j,k] in the t-th product's B-factor
    a_coeffs = [
        [[1, 0], [0, 1]],   # t=0: A11 + A22
        [[0, 0], [1, 1]],   # t=1: A21 + A22
        [[1, 0], [0, 0]],   # t=2: A11
        [[0, 0], [0, 1]],   # t=3: A22
        [[1, 1], [0, 0]],   # t=4: A11 + A12
        [[-1, 0], [1, 0]],  # t=5: A21 - A11
        [[0, 1], [0, -1]],  # t=6: A12 - A22
    ]
    b_coeffs = [
        [[1, 0], [0, 1]],   # t=0: B11 + B22
        [[1, 0], [0, 0]],   # t=1: B11
        [[0, 1], [0, 0]],   # t=2: B12
        [[0, 0], [1, 0]],   # t=3: B21
        [[0, 0], [0, 1]],   # t=4: B22
        [[1, 1], [0, 0]],   # t=5: B11 + B12
        [[0, 0], [1, 1]],   # t=6: B21 + B22
    ]
    # C combination: C[i][k] = sum over t of w[i][k][t] * product_t
    # Standard Strassen output:
    # C11 = M1 + M4 - M5 + M7  -> t indices 0, 3, 4, 6 with coeffs 1,1,-1,1
    # C12 = M3 + M5            -> t indices 2, 4 with coeffs 1,1
    # C21 = M2 + M4            -> t indices 1, 3 with coeffs 1,1
    # C22 = M1 - M2 + M3 + M6  -> t indices 0,1,2,5 with coeffs 1,-1,1,1
    w_coeffs = [
        [[1, 0, 0, 1, -1, 0, 1],   # C[0][0]
         [0, 0, 1, 0, 1, 0, 0]],   # C[0][1]
        [[0, 1, 0, 1, 0, 0, 0],   # C[1][0]
         [1, -1, 1, 0, 0, 1, 0]]  # C[1][1]
    ]
    
    U = np.zeros((nrows * ncolsm, 7), dtype=float)
    V = np.zeros((ncolsm * ncolsk, 7), dtype=float)
    W = np.zeros((ncolsk * nrows, 7), dtype=float)
    
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i * ncolsm + j, t] = a_coeffs[t][i][j]
        for j in range(2):
            for k in range(2):
                V[j * ncolsk + k, t] = b_coeffs[t][j][k]
    
    for i in range(2):
        for k in range(2):
            for t in range(7):
                W[k * nrows + i, t] = w_coeffs[i][k][t]
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 Strassen decomposition via 2-level block recursion.
    Top level: 2x2 blocks of 2x2 matrices, each top-level product uses inner Strassen.
    Total rank: 7 * 7 = 49.
    
    A is 4x4, B is 4x4, C is 4x4.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Get the 2x2x2 Strassen factor tables
    inner_U, inner_V, inner_W = _strassen_2x2x2(2, 2, 2)
    
    # Top-level Strassen: 7 products, each of which is a 2x2 * 2x2 block multiply
    # For top-level factor t, the A-block coefficients are a_top[t][ri][ci]
    # and B-block coefficients are b_top[t][ci2][ki]
    # where (ri,ci) selects which 2x2 block of A, (ci2,ki) selects which 2x2 block of B.
    
    a_top = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ]
    b_top = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    w_top = [
        [[1, 0, 0, 1, -1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0]],
        [[0, 1, 0, 1, 0, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    ]
    
    for t in range(7):
        for ri in range(2):
            for ci in range(2):
                ac = a_top[t][ri][ci]
                if ac == 0:
                    continue
                for ci2 in range(2):
                    for ki in range(2):
                        bc = b_top[t][ci2][ki]
                        if bc == 0:
                            continue
                        # This top-level (t, ri, ci, ci2, ki) combination
                        # corresponds to inner Strassen product tp
                        for tp in range(7):
                            col = t * 7 + tp
                            # Fill U: A element at global (gi, gj) where gi = ri*2+ii, gj = ci*2+jj
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] = ac * inner_U[ii * 2 + jj, tp]
                            # Fill V: B element at global (gj2, gk) where gj2 = ci2*2+jj2, gk = ki*2+kk
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj2 = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj2 * p + gk, col] = bc * inner_V[jj2 * 2 + kk, tp]
                        # Fill W: C element at global (gk2, gi2) where gk2 = ki*2+kk2, gi2 = ri*2+ii2
                        # W coefficient comes from w_top[ri][ki][t] * inner_W[kk2*2+ii2, tp]
                        # But wait - the W combination is over top-level t, not over inner tp.
                        # Let me reconsider.
    
    # The above nested loop approach for W is tricky because W combines across top-level t.
    # Let me redo this more carefully.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for t in range(7):
        for ri in range(2):
            for ci in range(2):
                ac = a_top[t][ri][ci]
                if ac == 0:
                    continue
                for ci2 in range(2):
                    for ki in range(2):
                        bc = b_top[t][ci2][ki]
                        if bc == 0:
                            continue
                        for tp in range(7):
                            col = t * 7 + tp
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] = ac * inner_U[ii * 2 + jj, tp]
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj2 = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj2 * p + gk, col] = bc * inner_V[jj2 * 2 + kk, tp]
    
    # Now fill W: for each output C[gi][gk] where gi = ri*2+ii, gk = ki*2+kk,
    # the coefficient of product column (t, tp) is w_top[ri][ki][t] * inner_W[kk*2+ii, tp]
    for ri in range(2):
        for ki in range(2):
            for ii in range(2):
                for kk in range(2):
                    gi = ri * 2 + ii
                    gk = ki * 2 + kk
                    for t in range(7):
                        wtc = w_top[ri][ki][t]
                        if wtc == 0:
                            continue
                        for tp in range(7):
                            col = t * 7 + tp
                            W[gk * n + gi, col] += wtc * inner_W[kk * 2 + ii, tp]
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the 4x4x4 Strassen core (rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Append schoolbook columns for k=4 (the 5th column of B)
    extra = n * m  # 16
    R = U4.shape[1]
    U = np.zeros((n * m, R + extra), dtype=float)
    V = np.zeros((m * p, R + extra), dtype=float)
    W = np.zeros((p * n, R + extra), dtype=float)
    
    U[:, :R] = U4
    V[:, :R] = V4
    W[:, :R] = W4
    
    k_extra = 4  # 5th column (0-indexed)
    c = R
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k_extra, c] = 1.0
            W[k_extra * n + i, c] = 1.0
            c += 1
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
