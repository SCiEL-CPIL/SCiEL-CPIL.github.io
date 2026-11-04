"""Matrix multiplication 4,4,8 — rank-minimal exact tensor decomposition.

Approach: structured random search over sparse sign matrices with a target
rank R, rank-slice reconstruction to enforce consistency, then greedy
rank reduction with exact verification. Falls back to schoolbook (128).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before the 120 s / 150 s kill

def _time_left():
    return _DEADLINE - time.time()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    # T[i,j,k] should equal sum_r U[i,r] V[j,r] W[k,r]
    n, m, p = T.shape
    rank = U.shape[1]
    # compute sum over r efficiently: (U @ V.T) gives [i,j] per r? do per r
    err = 0.0
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # outer contributions to T
        Tcalc = np.einsum('i,j,k->ijk', u, v, w)
        err += np.sum((Tcalc - T) ** 2)
    return err < tol

def _reconstruct(n, m, p, R, seed, sparsity=0.5):
    """Build a candidate rank-R decomposition by random sparse factors,
    then enforce slice-rank consistency via rank factorization."""
    rng = np.random.default_rng(seed)
    # random sparse sign matrices
    def rand_sparse(rows):
        A = np.zeros((rows, R))
        for r in range(R):
            idx = rng.choice(rows, size=max(1, int(rows * sparsity)), replace=False)
            A[idx, r] = rng.choice([-1.0, 1.0], size=len(idx))
        return A
    U = rand_sparse(n * m)
    V = rand_sparse(m * p)
    # reconstruct W so that each slice (over r) is consistent:
    # We want T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # For a fixed (i,j), the vector over k is (U[i,:] * V[j,:]) @ W.T  -> (U[i]*V[j]) dot W[k].
    # This is automatically consistent for ANY U,V,W. So instead we generate W randomly
    # and the tensor is defined; we then check if it matches target. To bias toward target,
    # reconstruct W from U,V to best match T.
    # For each k: T[:, :, k] = sum_r W[k,r] * (U[:,r] V[:,r])  =>  T[:,:,k] = W[k,:] @ M_k
    # where M_k[r, (i,j)] = U[i,r]*V[j,r]. Solve W[k,:] = T[:,:,k] @ pinv(M_k).
    M = np.zeros((R, n * m, m * p))
    for r in range(R):
        M[r] = np.outer(U[:, r], V[:, r])
    # M shape (R, nm, mp). For each k, T[:,:,k] (nm,mp) = W[k,:] (R) @ M[:,:,k]? 
    # Actually (W[k,:] @ M_k) where M_k[r, (i,j)] = U[i,r]*V[j,r] gives (i,j). Good.
    W = np.zeros((p * n, R))
    for k in range(p * n):
        M_k = M[:, :, k]  # (R, nm, mp)
        # T[:,:,k] = W[k,:] (R,) @ M_k  ->  T[:,:,k].T = M_k.T @ W[k,:]
        rhs = T_flat_k = None
        # solve least squares: M_k (R, N) with N=nm*mp, x=W[k,:] (R)
        Mk = M_k.reshape(R, -1)
        rhs = _TARGET[nm_idx, :, k].reshape(-1)  # nm*mp
        sol, *_ = np.linalg.lstsq(Mk.T, rhs, rcond=None)  # M_k.T is (N,R)
        W[k, :] = sol
    return U, V, W

# Precompute target
_T = None
def _get_target():
    global _T
    if _T is None:
        _T = _target_tensor(*DIMENSIONS)
    return _T

def construct():
    n, m, p = DIMENSIONS
    T = _get_target()
    nm, mp, pn = n * m, m * p, p * n

    best = _schoolbook()
    best_rank = n * m * p
    assert _verify(best["U"], best["V"], best["W"], T)

    # Search decreasing ranks; try to find exact decomposition.
    seed = 0
    for R in range(32, 129):  # from small to large; small is the prize
        if _time_left() < 20:
            break
        # number of random trials scaled by remaining time
        trials = 0
        while _time_left() > 15 and trials < 40:
            trials += 1
            rng = np.random.default_rng(seed)
            seed += 1
            # random sparse sign factors
            def rand_sparse(rows):
                A = np.zeros((rows, R))
                for r in range(R):
                    sz = max(1, int(rows * 0.5))
                    idx = rng.choice(rows, size=sz, replace=False)
                    A[idx, r] = rng.choice([-1.0, 1.0], size=sz)
                return A
            U = rand_sparse(nm)
            V = rand_sparse(mp)
            # reconstruct W to match T as closely as possible
            M = np.zeros((R, nm, mp))
            for r in range(R):
                M[r] = np.outer(U[:, r], V[:, r])
            W = np.zeros((pn, R))
            ok = True
            for k in range(pn):
                Mk = M[:, :, k].reshape(R, -1)
                rhs = T[:, :, k].reshape(-1)
                sol, *_ = np.linalg.lstsq(Mk.T, rhs, rcond=None)
                W[k, :] = sol
            # verify
            if _verify(U, V, W, T, tol=1e-7):
                # round to nearest half-integer to satisfy entry constraint
                Ur = np.rint(2 * U) / 2.0
                Vr = np.rint(2 * V) / 2.0
                Wr = np.rint(2 * W) / 2.0
                if _verify(Ur, Vr, Wr, T, tol=1e-7):
                    best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    best_rank = R
                    # found a valid one at this rank; try to go lower
                    break
        # if found at this R, continue to try lower R (loop goes up, so break)
        if best_rank == R:
            # keep trying smaller ranks by restarting loop from smaller
            pass

    # Also try a direct greedy-reduction from schoolbook
    if _time_left() > 30:
        U = np.array(best["U"], dtype=float)
        V = np.array(best["V"], dtype=float)
        W = np.array(best["W"], dtype=float)
        # Greedy: drop columns with smallest energy, re-solve.
        for _ in range(min(40, int(_time_left() / 5))):
            if _time_left() < 10:
                break
            R = U.shape[1]
            if R <= 32:
                break
            # energy of each column
            energies = []
            for r in range(R):
                e = np.sum((U[:, r] * V[:, r] * W[:, r]) ** 2)
                energies.append(e)
            energies = np.array(energies)
            drop = int(np.argmin(energies))
            U2 = np.delete(U, drop, axis=1)
            V2 = np.delete(V, drop, axis=1)
            W2 = np.delete(W, drop, axis=1)
            # re-solve W to reduce error
            M = np.zeros((R - 1, nm, mp))
            for r in range(R - 1):
                M[r] = np.outer(U2[:, r], V2[:, r])
            Wnew = np.zeros((pn, R - 1))
            for k in range(pn):
                Mk = M[:, :, k].reshape(R - 1, -1)
                rhs = T[:, :, k].reshape(-1)
                sol, *_ = np.linalg.lstsq(Mk.T, rhs, rcond=None)
                Wnew[k, :] = sol
            if _verify(U2, V2, Wnew, T, tol=1e-7):
                U, V, W = U2, V2, Wnew
                Ur = np.rint(2 * U) / 2.0
                Vr = np.rint(2 * V) / 2.0
                Wr = np.rint(2 * W) / 2.0
                if _verify(Ur, Vr, Wr, T, tol=1e-7):
                    best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    best_rank = U.shape[1]

    return best

# precompute target at module level for speed in construct
_T = _target_tensor(*DIMENSIONS)
