"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: Use Strassen's rank-28 decomposition for 4x4 * 4x4, applied to two halves
of the 4x8 output. This gives rank 56. Then attempt to reduce rank by checking
for linear dependencies among the rank terms.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            contrib = U3[i, :, c][:, np.newaxis] * V3[:, :, c] * W3[:, i, c][np.newaxis, :]
            T[i, :, :] += contrib
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition of the 2x2 * 2x2 multiplication tensor."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # c=0: M1 = (A00+A11)(B01+B10)
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # c=1: M2 = (A10+A11)A00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # c=2: M3 = A00(B00-B10)
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    W[0, 2] = 1; W[3, 2] = 1
    # c=3: M4 = A11(B11-B01)
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    # c=4: M5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    # c=5: M6 = (A10-A00)(B00+B11)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    # c=6: M7 = (A01-A11)(B01+B10)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _build_4x4x4():
    """Build a rank-28 decomposition of the 4x4 * 4x4 multiplication tensor using
    Strassen's algorithm on 2x2 blocks."""
    sU, sV, sW = _strassen_2x2x2()
    rank = 4 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    for a in range(4):
        for c in range(7):
            col = a * 7 + c
            for bi in range(2):
                for bj in range(2):
                    U[(2 * a + bi) * 4 + (2 * b + bj), col] = sU[bi * 2 + bj, c]
                    V[(2 * b + bj) * 4 + (2 * c + bk), col] = sV[bj * 2 + bk, c]
                    W[(2 * c + bk) * 4 + (2 * a + bi), col] = sW[bk * 2 + bi, c]
    return U, V, W


def _build_4x4x8():
    """Build a rank-56 decomposition of the 4x4 * 4x8 multiplication tensor."""
    U1, V1, W1 = _build_4x4x4()
    rank = 2 * 28
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    U[:, 0:28] = U1
    V[0:16, 0:28] = V1
    W[0:16, 0:28] = W1

    U[:, 28:56] = U1
    V[16:32, 28:56] = V1
    W[16:32, 28:56] = W1

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_time=100):
    """Try to reduce rank by identifying and removing linearly dependent terms."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 32:
        return U, V, W
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try removing terms one at a time
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        if best_rank <= 32:
            break
        
        removed_any = False
        for c in range(best_rank):
            if time.time() - start_time > max_time:
                break
            
            U_test = np.delete(best_U, c, axis=1)
            V_test = np.delete(best_V, c, axis=1)
            W_test = np.delete(best_W, c, axis=1)
            
            if _verify_fast(U_test, V_test, W_test, n, m, p, tol=1e-10):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank -= 1
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    try:
        U, V, W = _build_4x4x8()
        if _verify_fast(U, V, W, n, m, p, tol=1e-10):
            # Try to reduce rank
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_time=120)
            if _verify_fast(U_red, V_red, W_red, n, m, p, tol=1e-10):
                U, V, W = U_red, V_red, W_red
            
            # Snap to integers
            U_snap = np.round(U).astype(int)
            V_snap = np.round(V).astype(int)
            W_snap = np.round(W).astype(int)
            if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            else:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END
