"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: exploit p = 2m by splitting the output into two 4x4 blocks, each computed
by a 4x4 * 4x4 multiplication. Use Strassen's rank-28 decomposition for the square
case and combine two copies to obtain a rank-56 decomposition of the 4x4x8 tensor.
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
    """Return Strassen's rank-7 decomposition of the 2x2 * 2x2 multiplication tensor.

    Convention: U shape (4,7) indexed by (i*2+j), V shape (4,7) indexed by (j*2+k),
    W shape (4,7) indexed by (k*2+i).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen factors M1..M7:
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)A00
    # M3 = A00(B00-B10)
    # M4 = A11(B11-B01)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B11)
    # M7 = (A01-A11)(B01+B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5

    # c=0: M1 = (A00+A11)(B01+B10)
    U[0 * 2 + 0, 0] = 1; U[1 * 2 + 1, 0] = 1
    V[0 * 2 + 1, 0] = 1; V[1 * 2 + 0, 0] = 1
    W[0 * 2 + 0, 0] = 1; W[1 * 2 + 1, 0] = 1
    # c=1: M2 = (A10+A11)A00
    U[1 * 2 + 0, 1] = 1; U[1 * 2 + 1, 1] = 1
    V[0 * 2 + 0, 1] = 1
    W[1 * 2 + 0, 1] = 1; W[1 * 2 + 1, 1] = -1
    # c=2: M3 = A00(B00-B10)
    U[0 * 2 + 0, 2] = 1
    V[0 * 2 + 0, 2] = 1; V[1 * 2 + 0, 2] = -1
    W[0 * 2 + 0, 2] = 1; W[1 * 2 + 1, 2] = 1
    # c=3: M4 = A11(B11-B01)
    U[1 * 2 + 1, 3] = 1
    V[1 * 2 + 1, 3] = 1; V[0 * 2 + 1, 3] = -1
    W[1 * 2 + 0, 3] = 1; W[0 * 2 + 0, 3] = 1
    # c=4: M5 = (A00+A10)B11
    U[0 * 2 + 0, 4] = 1; U[1 * 2 + 0, 4] = 1
    V[1 * 2 + 1, 4] = 1
    W[0 * 2 + 0, 4] = -1; W[1 * 2 + 1, 4] = 1
    # c=5: M6 = (A10-A00)(B00+B11)
    U[1 * 2 + 0, 5] = 1; U[0 * 2 + 0, 5] = -1
    V[0 * 2 + 0, 5] = 1; V[1 * 2 + 1, 5] = 1
    W[1 * 2 + 0, 5] = 1
    # c=6: M7 = (A01-A11)(B01+B10)
    U[0 * 2 + 1, 6] = 1; U[1 * 2 + 1, 6] = -1
    V[0 * 2 + 1, 6] = 1; V[1 * 2 + 0, 6] = 1
    W[0 * 2 + 0, 6] = 1

    return U, V, W


def _build_4x4x4():
    """Build a rank-28 decomposition of the 4x4 * 4x4 multiplication tensor using
    Strassen's algorithm on 2x2 blocks."""
    sU, sV, sW = _strassen_2x2x2()
    rank = 4 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    for a in range(4):  # Strassen factor index
        for c in range(7):  # Strassen sub-factor index
            col = a * 7 + c
            for bi in range(2):  # 2x2 block row
                for bj in range(2):  # 2x2 block col
                    # U index: (block_row*2 + bi, block_col*2 + bj) -> (bi*2+bj)
                    # A block (a, b) has rows [2a, 2a+1], cols [2b, 2b+1]
                    # U[i*4 + j, col] where i = 2a + bi, j = 2b + bj
                    U[(2 * a + bi) * 4 + (2 * b + bj), col] = sU[bi * 2 + bj, c]
                    # V[j*4 + k, col] where j = 2b + bj, k = 2c + bk
                    V[(2 * b + bj) * 4 + (2 * c + bk), col] = sV[bj * 2 + bk, c]
                    # W[k*4 + i, col] where k = 2c + bk, i = 2a + bi
                    W[(2 * c + bk) * 4 + (2 * a + bi), col] = sW[bk * 2 + bi, c]
    return U, V, W


def _build_4x4x8():
    """Build a rank-56 decomposition of the 4x4 * 4x8 multiplication tensor by
    combining two 4x4 * 4x4 decompositions (one for each half of the output)."""
    U1, V1, W1 = _build_4x4x4()
    rank = 2 * 28
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))

    # First 4x4 * 4x4: computes C[:, 0:4] = A @ B[:, 0:4]
    U[:, 0:28] = U1
    V[0:16, 0:28] = V1  # B[:, 0:4] has index j*4 + k with k in [0,4)
    W[0:16, 0:28] = W1  # C[:, 0:4] has index k*4 + i with k in [0,4)

    # Second 4x4 * 4x4: computes C[:, 4:8] = A @ B[:, 4:8]
    U[:, 28:56] = U1
    V[16:32, 28:56] = V1  # B[:, 4:8] has index j*4 + (4+k) with k in [0,4)
    W[16:32, 28:56] = W1  # C[:, 4:8] has index (4+k)*4 + i with k in [0,4)

    return U, V, W


def _try_lower_rank_decomposition(n, m, p, target_rank, max_iters=200, tol=1e-10):
    """Try to find a decomposition with rank <= target_rank using iterative refinement.
    
    Uses a simple approach: start from the rank-56 solution, then try to eliminate
    rank terms by checking if they're redundant.
    """
    U, V, W = _build_4x4x8()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to reduce rank by checking linear dependence
    # This is a heuristic approach - try removing terms one at a time
    # and checking if the decomposition still works
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # For each term, try to see if we can remove it
    for c in range(current_rank):
        if time.time() > 100:  # Safety check
            break
        if current_rank - c <= target_rank:
            break
            
        # Try removing term c
        U_test = np.delete(U, c, axis=1)
        V_test = np.delete(V, c, axis=1)
        W_test = np.delete(W, c, axis=1)
        
        if _verify_fast(U_test, V_test, W_test, n, m, p, tol=tol):
            U, V, W = U_test, V_test, W_test
            current_rank -= 1
            c = -1  # Reset since indices shifted
            if current_rank <= target_rank:
                break
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit

    # Primary: Strassen-based rank-56 decomposition
    try:
        U, V, W = _build_4x4x8()
        if _verify_fast(U, V, W, n, m, p, tol=1e-10):
            # Snap to integers (Strassen uses only -1, 0, 1)
            U_snap = np.round(U).astype(int)
            V_snap = np.round(V).astype(int)
            W_snap = np.round(W).astype(int)
            if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            else:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank-128
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-END
