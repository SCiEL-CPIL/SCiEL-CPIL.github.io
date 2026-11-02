"""Fresh design: search for a low-rank exact tensor decomposition of (2,2,13)
matrix multiplication, with a guaranteed-valid schoolbook fallback and an
explicit deadline guard."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 13]
DEADLINE = 150.0
SAFETY_MARGIN = 6.0


def _build_tensor(n, m, p):
    """Tensor T[k, i, j] = 1 if i,j,k all in range (matmul tensor)."""
    T = np.zeros((p, n, m), dtype=np.float64)
    for k in range(p):
        for i in range(n):
            for j in range(m):
                T[k, i, j] = 1.0
    return T


def _tensor_to_2d(T):
    """Flatten T[k,i,j] -> matrix with row (k*n+i), col j."""
    n = T.shape[1]
    p = T.shape[0]
    m = T.shape[2]
    return T.reshape(p * n, m)


def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _rank_of(A, B, C):
    """Rank of decomposition = number of all-zero columns removed from common
    structure; here we just count columns that are non-zero in all of U,V,W."""
    # A rank decomposition uses the same number of terms = number of columns.
    return A.shape[1]


def _validate(U, V, W, n, m, p, tol=1e-8):
    """Check that sum_c U[:,c] V[:,c] W[:,c] == T exactly (within tol)."""
    T = _build_tensor(n, m, p)
    T2 = _tensor_to_2d(T)
    # Contract: for each column c, add outer(U[:,c], V[:,c]) * W[:,c]
    # Build matrix M with shape (p*n, m): M = sum_c W[:,c][:,None] * (U[:,c] V[:,c])
    # But U is (n*m, rank), V is (m*p, rank). We need to map indices.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Term c contributes to T[k,i,j] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    # Reconstruct: for each c, build matrix A_c (n*m, m*p) = U[:,c] outer V[:,c].
    # A_c[(i*m+j),(j*p+k)] = U[i*m+j,c]*V[j*p+k,c].
    # Then T[k,i,j] = sum_c A_c[(i*m+j),(j*p+k)] * W[k*n+i,c].
    # Build full M[k*n+i, j] = sum_c W[k*n+i,c] * (sum_j? ) hmm need care.
    # Let's just brute force accumulate.
    M = np.zeros((p * n, m), dtype=np.float64)
    for c in range(U.shape[1]):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # uc index: i*m+j ; vc index: j*p+k ; wc index: k*n+i
        # For each i,j,k: M[k*n+i, j] += uc[i*m+j]*vc[j*p+k]*wc[k*n+i]
        # Vectorize over j:
        for i in range(n):
            for j in range(m):
                uval = uc[i * m + j]
                if uval == 0.0:
                    continue
                # vc[j*p : (j+1)*p] are the k values for this j
                vcj = vc[j * p:(j + 1) * p]
                # wc[k*n + i] for k in range(p): indices i, i+n, i+2n,...
                wci = wc[i::n]
                # contribution to M[k*n+i, j] for each k
                M[i::n, j] += uval * (vcj * wci)
    return np.max(np.abs(M - T2)) < tol


def _try_gradient_search(n, m, p, rank, rng, time_limit):
    """Try to find a rank-`rank` decomposition via gradient descent on
    the residual ||M - T||^2. Returns (U,V,W) if found within tol, else None."""
    T = _build_tensor(n, m, p)
    T2 = _tensor_to_2d(T)

    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    # Scale down
    U *= 0.1
    V *= 0.1
    W *= 0.1

    lr = 0.02
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    start = time.time()
    it = 0
    # Precompute index maps for gradient
    # M[k*n+i, j] = sum_c W[k*n+i,c] * sum_j? Actually:
    # M[k*n+i, j] = sum_c W[k*n+i,c] * (sum over j' of U[i*m+j',c]*V[j'*p+k,c])
    # Let B_c[k, i, j] = U[i*m+j,c] * V[j*p+k,c]  (n*m*p entries per c)
    # Then M[k*n+i, j] = sum_c W[k*n+i,c] * B_c[k,i,j]
    # Residual R = M - T2.
    # dR/dW[k*n+i,c] = B_c[k,i,j]
    # dR/dU[i*m+j,c] = W[k*n+i,c] * V[j*p+k,c]  (for each k)
    #   -> grad_U[i*m+j,c] = sum_k W[k*n+i,c]*V[j*p+k,c]*R[k*n+i,j]
    # dR/dV[j*p+k,c] = W[k*n+i,c] * U[i*m+j,c]
    #   -> grad_V[j*p+k,c] = sum_i W[k*n+i,c]*U[i*m+j,c]*R[k*n+i,j]

    # Precompute for speed: for each c, B_c as 3D (p,n,m)
    # We'll recompute B each iteration.
    def contract(U, V, W):
        M = np.zeros((p * n, m))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    uval = uc[i * m + j]
                    if uval == 0.0:
                        continue
                    vcj = vc[j * p:(j + 1) * p]
                    wci = wc[i::n]
                    M[i::n, j] += uval * (vcj * wci)
        return M

    def grad_step(U, V, W, lr):
        M = contract(U, V, W)
        R = M - T2
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # gW[k*n+i,c] = B_c[k,i,j] = uc[i*m+j]*vc[j*p+k]
            # accumulate over j
            for i in range(n):
                for j in range(m):
                    uval = uc[i * m + j]
                    if uval == 0.0:
                        continue
                    vcj = vc[j * p:(j + 1) * p]
                    # gW[i::n, c] += uval * vcj
                    gW[i::n, c] += uval * vcj
                    # gU[i*m+j,c] = sum_k wc[k*n+i]*vc[j*p+k]*R[k*n+i,j]
                    wci = wc[i::n]
                    gU[i * m + j, c] += np.dot(wci * vcj, R[i::n, j])
                    # gV[j*p+k,c] = sum_i wc[k*n+i]*uc[i*m+j]*R[k*n+i,j]
                    # gV[j*p : (j+1)*p, c] += uc[i*m+j] * (wc[i::n] * R[i::n, j])
                    gV[j * p:(j + 1) * p, c] += uval * (wc[i::n] * R[i::n, j])
        return M, R, gU, gV, gW

    # Adam-like
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    while True:
        if time.time() - start > time_limit:
            break
        M, R, gU, gV, gW = grad_step(U, V, W, lr)
        err = np.max(np.abs(R))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-8:
            return best_U, best_V, best_W
        it += 1
        t1 = 1 - b1 ** it
        t2 = 1 - b2 ** it
        mU = b1 * mU + (1 - b1) * gU; vU = b2 * vU + (1 - b2) * (gU ** 2)
        mV = b1 * mV + (1 - b1) * gV; vV = b2 * vV + (1 - b2) * (gV ** 2)
        mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * (gW ** 2)
        mUh = mU / t1; vUh = vU / t2
        mVh = mV / t1; vVh = vV / t2
        mWh = mW / t1; vWh = vW / t2
        U -= lr * mUh / (np.sqrt(vUh) + eps)
        V -= lr * mVh / (np.sqrt(vVh) + eps)
        W -= lr * mWh / (np.sqrt(vWh) + eps)
        # decay lr slowly
        lr = max(0.002, lr * 0.999)

    # Final check
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    # Guaranteed fallback
    U0, V0, W0 = _make_schoolbook(n, m, p)

    # Try to find a better (lower rank) decomposition via gradient search.
    # Lower bound for matmul (2,2,13): tensor rank at least max(n*m, m*p, p*n)
    # = max(4, 26, 26) = 26. So rank < 52 is the target.
    best = (U0.shape[1], U0, V0, W0)

    rng = np.random.default_rng(12345)

    # Try a few candidate ranks below schoolbook.
    for rank in [40, 36, 32, 30, 28, 26]:
        if time.time() > deadline:
            break
        rem = deadline - time.time()
        # Allocate a chunk of time per rank
        chunk = min(rem * 0.25, 20.0)
        if chunk < 2.0:
            break
        res = _try_gradient_search(n, m, p, rank, rng, chunk)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p, tol=1e-8):
                if rank < best[0]:
                    best = (rank, U, V, W)
                # Found a valid one; we could try even lower, but keep going.

    if time.time() > deadline:
        # Return best found so far
        r, U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    r, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
