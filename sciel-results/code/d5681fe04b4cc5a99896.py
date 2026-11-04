import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(S, np.ones((n, m, p)), atol=tol):
        return False
    # Check half-integer constraint
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
        return False
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
        return False
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
        return False
    return True

def _snap_to_half_int(U, V, W):
    """Snap matrices to nearest half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _als_3x3x3(U, V, W, rank, deadline):
    """One ALS step for 3x3x3 tensor decomposition."""
    n = m = p = 3
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except:
                pass
    U = U_new

    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except:
                pass
    V = V_new

    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except:
                pass
    W = W_new

    return U, V, W

def _err_3x3x3(U, V, W, rank):
    """Compute error for 3x3x3 decomposition."""
    n = m = p = 3
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.sum((S - 1.0)**2)

def _try_rank_3x3x3(rank, seed, deadline, max_iters=200):
    """Try to find a rank-r decomposition for 3x3x3 using ALS."""
    rng = np.random.RandomState(seed)
    n = m = p = 3
    # Initialize with random values, scaled appropriately
    U = rng.randn(n*m, rank) * 0.5
    V = rng.randn(m*p, rank) * 0.5
    W = rng.randn(p*n, rank) * 0.5

    best_err = float('inf')
    best_U, best_V, best_W = None, None, None

    for it in range(max_iters):
        if time.time() > deadline:
            break
        U, V, W = _als_3x3x3(U, V, W, rank, deadline)
        err = _err_3x3x3(U, V, W, rank)

        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_err < 1e-12:
            break

    if best_err < 1e-10:
        U_r, V_r, W_r = _snap_to_half_int(best_U, best_V, best_W)
        # Verify
        U_3d = U_r.reshape(3, 3, rank)
        V_3d = V_r.reshape(3, 3, rank)
        W_3d = W_r.reshape(3, 3, rank)
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if np.allclose(S, np.ones((3,3,3)), atol=1e-8):
            return U_r, V_r, W_r

    return None, None, None

def _construct_3x3x6_from_3x3x3(U19, V19, W19, rank):
    """
    Construct a 3x3x6 decomposition from a 3x3x3 decomposition.
    C = A * B where B = [B1 | B2], each 3x3.
    C = A*B1 + A*B2.
    U factor is shared (same A), V and W are split into two blocks.
    """
    n, m, p = 3, 3, 6
    total_rank = 2 * rank

    U38 = np.zeros((n*m, total_rank))
    V38 = np.zeros((m*p, total_rank))
    W38 = np.zeros((p*n, total_rank))

    # First block: k=0,1,2
    U38[:, 0:rank] = U19
    for j in range(m):
        for k in range(3):
            V38[j*p + k, 0:rank] = V19[j*3 + k, :]
    for k in range(3):
        for i in range(n):
            W38[k*n + i, 0:rank] = W19[k*3 + i, :]

    # Second block: k=3,4,5
    U38[:, rank:2*rank] = U19
    for j in range(m):
        for k in range(3):
            V38[j*p + (k+3), rank:2*rank] = V19[j*3 + k, :]
    for k in range(3):
        for i in range(n):
            W38[(k+3)*n + i, rank:2*rank] = W19[k*3 + i, :]

    # Verify
    if _verify(U38, V38, W38, n, m, p, tol=1e-8):
        return U38, V38, W38

    # Try snapping
    U_r, V_r, W_r = _snap_to_half_int(U38, V38, W38)
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-8):
        return U_r, V_r, W_r

    return None, None, None

def _als_3x3x6(U, V, W, n, m, p, rank, deadline):
    """One ALS step for 3x3x6 tensor decomposition."""
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except:
                pass
    U = U_new

    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except:
                pass
    V = V_new

    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except:
                pass
    W = W_new

    return U, V, W

def _err_3x3x6(U, V, W, n, m, p, rank):
    """Compute error for 3x3x6 decomposition."""
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.sum((S - 1.0)**2)

def _try_rank_3x3x6(rank, seed, deadline, max_iters=100):
    """Try to find a rank-r decomposition for 3x3x6 using ALS."""
    rng = np.random.RandomState(seed)
    n, m, p = 3, 3, 6
    U = rng.randn(n*m, rank) * 0.5
    V = rng.randn(m*p, rank) * 0.5
    W = rng.randn(p*n, rank) * 0.5

    best_err = float('inf')
    best_U, best_V, best_W = None, None, None

    for it in range(max_iters):
        if time.time() > deadline:
            break
        U, V, W = _als_3x3x6(U, V, W, n, m, p, rank, deadline)
        err = _err_3x3x6(U, V, W, n, m, p, rank)

        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_err < 1e-12:
            break

    if best_err < 1e-10:
        U_r, V_r, W_r = _snap_to_half_int(best_U, best_V, best_W)
        if _verify(U_r, V_r, W_r, n, m, p, tol=1e-8):
            return U_r, V_r, W_r

    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Fallback: schoolbook
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = n * m * p

    # Strategy 1: Find low-rank 3x3x3, then construct 3x3x6 as 2*rank
    # Try ranks from 19 upward (known minimum for 3x3x3 is 19)
    for r in range(19, 25):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            U19, V19, W19 = _try_rank_3x3x3(r, seed, deadline, max_iters=150)
            if U19 is not None:
                U38, V38, W38 = _construct_3x3x6_from_3x3x3(U19, V19, W19, r)
                if U38 is not None:
                    total_rank = 2 * r
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_U, best_V, best_W = U38, V38, W38
                    found = True
                    break
        if found:
            break

    # Strategy 2: Direct 3x3x6 search at ranks below current best
    # Try ranks from best_rank-1 down to 20
    for R in range(min(best_rank - 1, 40), 19, -1):
        if time.time() > deadline - 5:
            break
        if R >= best_rank:
            continue
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            U, V, W = _try_rank_3x3x6(R, seed, deadline, max_iters=80)
            if U is not None:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        if found:
            break

    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_fb, V_fb, W_fb
        best_rank = n * m * p

    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
