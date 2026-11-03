"""Matrix multiplication 3x3, 3x4 - rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_strassen_rank34():
    """Build rank-34 decomposition using two Strassen 2x2x2 blocks + schoolbook."""
    n, m, p = 3, 3, 4
    rank = 34
    U = np.zeros((9, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    col = 0
    
    # Strassen 2x2x2 block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*3+j -> (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # V indices: j*4+k -> (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # W indices: k*3+i -> (0,0)=0, (1,0)=3, (0,1)=1, (1,1)=4
    strassen1 = [
        # M1=(A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        ({0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 4: 1}),
        # M2=(A10+A11)B00 -> C10+=M2
        ({3: 1, 4: 1}, {0: 1}, {1: 1}),
        # M3=A00(B10-B11) -> C01+=M3
        ({0: 1}, {4: 1, 5: -1}, {3: 1}),
        # M4=A11(B01-B00) -> C10+=M4, C11+=M4
        ({4: 1}, {1: 1, 0: -1}, {1: 1, 4: 1}),
        # M5=(A00+A01)B11 -> C01+=M5
        ({0: 1, 1: 1}, {5: 1}, {3: 1}),
        # M6=(A10-A00)(B00+B01) -> C11+=M6
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1}),
        # M7=(A01-A11)(B10+B11) -> C00+=M7
        ({1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1}),
    ]
    
    for a_part, b_part, w_contribs in strassen1:
        for u_idx, val in a_part.items():
            U[u_idx, col] += val
        for v_idx, val in b_part.items():
            V[v_idx, col] += val
        for w_idx, val in w_contribs.items():
            W[w_idx, col] += val
        col += 1
    
    # Strassen 2x2x2 block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # V indices: j*4+k -> (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # W indices: k*3+i -> (2,0)=6, (3,0)=9, (2,1)=7, (3,1)=10
    strassen2 = [
        ({0: 1, 4: 1}, {2: 1, 7: 1}, {6: 1, 10: 1}),
        ({3: 1, 4: 1}, {2: 1}, {7: 1}),
        ({0: 1}, {6: 1, 7: -1}, {9: 1}),
        ({4: 1}, {3: 1, 2: -1}, {7: 1, 10: 1}),
        ({0: 1, 1: 1}, {7: 1}, {9: 1}),
        ({3: 1, 0: -1}, {2: 1, 3: 1}, {10: 1}),
        ({1: 1, 4: -1}, {6: 1, 7: 1}, {6: 1}),
    ]
    
    for a_part, b_part, w_contribs in strassen2:
        for u_idx, val in a_part.items():
            U[u_idx, col] += val
        for v_idx, val in b_part.items():
            V[v_idx, col] += val
        for w_idx, val in w_contribs.items():
            W[w_idx, col] += val
        col += 1
    
    # Remaining 20 schoolbook terms
    for i in range(3):
        for j in range(3):
            for k in range(4):
                if i in (0, 1) and j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i * 3 + j, col] = 1
                V[j * 4 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    
    assert col == 34, f"Expected 34, got {col}"
    return U, V, W


def _verify(U, V, W, tol=1e-10):
    n, m, p = 3, 3, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _rationalize(mat, tol=1e-9):
    result = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if abs(val) < tol:
                result[i, j] = 0.0
                continue
            found = False
            for denom in range(1, 201):
                num = round(val * denom)
                if abs(val - num / denom) < tol:
                    result[i, j] = num / denom
                    found = True
                    break
            if not found:
                result[i, j] = val
    return result


def construct():
    t0 = time.time()
    deadline = t0 + 140
    
    n, m, p = 3, 3, 4
    
    U34, V34, W34 = _build_strassen_rank34()
    if _verify(U34, V34, W34):
        best_rank = 34
        best_U, best_V, best_W = U34.copy(), V34.copy(), W34.copy()
    else:
        best_rank = 36
        best_U = np.zeros((9, 36))
        best_V = np.zeros((12, 36))
        best_W = np.zeros((12, 36))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    best_U[i * 3 + j, col] = 1
                    best_V[j * 4 + k, col] = 1
                    best_W[k * 3 + i, col] = 1
                    col += 1
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
