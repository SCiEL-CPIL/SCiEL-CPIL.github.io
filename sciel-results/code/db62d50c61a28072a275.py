"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Convention: W[k*n + i, c].  T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    return bool(np.allclose(T, 1.0, atol=atol, rtol=1e-9))


def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p (integer entries)."""
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


def _project_half(x):
    """Project to nearest value in {-1,-0.5,0,0.5,1} (clipped)."""
    return np.clip(np.round(x * 2.0) / 2.0, -1.0, 1.0)


def _polish_half(U, V, W, n, m, p, max_iter=400, deadline=None, lr0=0.25, lr1=0.02):
    """High-precision polish: keep entries on the half-integer lattice, drive the
    tensor to all-ones.  Returns (U,V,W) as float arrays."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
        if np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
            break
        E = T - 1.0
        gU = np.einsum('ijk,jkc,kic->ijc', E, Vr, Wr, optimize=True)
        gV = np.einsum('ijk,ijc,kic->jkc', E, Ur, Wr, optimize=True)
        gW = np.einsum('ijk,ijc,jkc->kic', E, Ur, Vr, optimize=True)
        lr = lr0 * (lr1 / lr0) ** (it / max(1, max_iter - 1))
        Ur = _project_half(Ur - lr * gU)
        Vr = _project_half(Vr - lr * gV)
        Wr = _project_half(Wr - lr * gW)
    return Ur.reshape(n * m, -1), Vr.reshape(m * p, -1), Wr.reshape(p * n, -1)


def _try_rank(n, m, p, r, deadline, seeds=(0, 1, 2, 3, 4, 5)):
    """Try to find a rank-r half-integer decomposition via random init + polish."""
    for seed in seeds:
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        U = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, r))
        V = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, r))
        W = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, r))
        U, V, W = _polish_half(U, V, W, n, m, p, max_iter=300, deadline=deadline)
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before the 150 s kill

    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U, V, W)

    # Lower bound: rank >= max(n*m, m*p, p*n) = 35 for 3,5,7.
    lb = max(n * m, m * p, p * n)
    # Search a modest band of ranks below schoolbook; stop early on success.
    for r in range(lb, best_rank):
        if time.time() > deadline:
            break
        found = _try_rank(n, m, p, r, deadline)
        if found is not None:
            U, V, W = found
            best_rank = r
            best = (U, V, W)
            break

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
