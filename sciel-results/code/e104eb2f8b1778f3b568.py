"""Matrix multiplication 3x5x6 - structured Strassen decomposition with dependency search."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
N, M, P = 3, 5, 6
A_ROWS = N * M  # 15
B_ROWS = M * P  # 30
C_ROWS = P * N  # 18


def _build_T():
    T = np.zeros((A_ROWS, B_ROWS, C_ROWS))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T


_T = _build_T()


def _schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(A_ROWS)]
    V = [[0] * rank for _ in range(B_ROWS)]
    W = [[0] * rank for _ in range(C_ROWS)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _matmul(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _residual(U, V, W):
    return np.abs(_matmul(U, V, W) - _T).max()


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen (rank 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    T[i,j,k] = sum_c U[i*2+j,c] V[j*2+k,c] W[k*2+i,c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1

    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1

    return U, V, W


def _build_strassen_decomp():
    """Build a valid decomposition for 3x5 @ 5x6 using Strassen + rank-1 terms."""
    sU, sV, sW = _strassen_2x2x2()
    
    terms = []  # List of (u, v, w) vectors
    
    # A (3x5), B (5x6)
    # Split A: rows 0-1 (top), row 2 (bottom)
    # Split B: cols 0-1, 2-3, 4-5
    
    for bblock in range(3):
        bcol0 = bblock * 2  # First col of this B block
        
        # Top part: A[0:2, :] (2x5) @ B[:, bcol0:bcol0+2] (5x2) -> C[0:2, bcol0:bcol0+2] (2x2)
        # Split A_top: cols 0-1 (A1a, 2x2) and cols 2-4 (A1b, 2x3)
        # Split B_block: rows 0-1 (B1a, 2x2) and rows 2-4 (B1b, 3x2)
        
        # A1a @ B1a: 2x2 @ 2x2, Strassen rank 7
        # A1a[i, j] -> A row i*5 + j, for i in {0,1}, j in {0,1}
        # B1a[j, k] -> B row j*6 + (bcol0 + k), for j in {0,1}, k in {0,1}
        # C[i, k] -> C row (bcol0 + k)*3 + i, for i in {0,1}, k in {0,1}
        for c in range(7):
            u = np.zeros(A_ROWS)
            v = np.zeros(B_ROWS)
            w = np.zeros(C_ROWS)
            for row in range(4):
                if sU[row, c] != 0:
                    i = row // 2
                    j = row % 2
                    u[i * 5 + j] = sU[row, c]
                if sV[row, c] != 0:
                    j = row // 2
                    k = row % 2
                    v[j * 6 + bcol0 + k] = sV[row, c]
                if sW[row, c] != 0:
                    k = row // 2
                    i = row % 2
                    w[(bcol0 + k) * 3 + i] = sW[row, c]
            terms.append((u, v, w))
        
        # A1b @ B1b: 2x3 @ 3x2, rank-1 terms (6 terms)
        # A1b[i, j] -> A row i*5 + (j+2), for i in {0,1}, j in {0,1,2}
        # B1b[j, k] -> B row (j+2)*6 + (bcol0 + k), for j in {0,1,2}, k in {0,1}
        # C[i, k] -> C row (bcol0 + k)*3 + i
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u = np.zeros(A_ROWS)
                    v = np.zeros(B_ROWS)
                    w = np.zeros(C_ROWS)
                    u[i * 5 + j + 2] = 1
                    v[(j + 2) * 6 + bcol0 + k] = 1
                    w[(bcol0 + k) * 3 + i] = 1
                    terms.append((u, v, w))
        
        # Bottom part: A[2, :] (1x5) @ B[:, bcol0:bcol0+2] (5x2) -> C[2, bcol0:bcol0+2] (1x2)
        # A[2, j] -> A row 10 + j
        # B[j, k] -> B row j*6 + bcol0 + k
        # C[k] -> C row (bcol0 + k)*3 + 2
        for j in range(5):
            for k in range(2):
                u = np.zeros(A_ROWS)
                v = np.zeros(B_ROWS)
                w = np.zeros(C_ROWS)
                u[10 + j] = 1
                v[j * 6 + bcol0 + k] = 1
                w[(bcol0 + k) * 3 + 2] = 1
                terms.append((u, v, w))
    
    rank = len(terms)
    U = np.column_stack([t[0] for t in terms])
    V = np.column_stack([t[1] for t in terms])
    W = np.column_stack([t[2] for t in terms])
    
    # Verify
    recon = _matmul(U, V, W)
    res = np.abs(recon - _T).max()
    if res > 1e-9:
        return None
    
    return U, V, W, rank


def _verify(U, V, W, tol=1e-9):
    recon = _matmul(U, V, W)
    return np.abs(recon - _T).max() < tol


def _build_result(U, V, W):
    rank = U.shape[1]
    U_list = [[float(x) for x in u] for u in U]
    V_list = [[float(x) for x in v] for v in V]
    W_list = [[float(x) for x in w] for w in W]
    return {"U": U_list, "V": V_list, "W": W_list}


def construct():
    deadline = time.time() + 140
    
    result = _build_strassen_decomp()
    if result is not None:
        U, V, W, rank = result
        if _verify(U, V, W):
            return _build_result(U, V, W)
    
    return _schoolbook()
