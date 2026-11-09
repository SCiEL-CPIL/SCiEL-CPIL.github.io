import time
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 4, 5]
START = time.time()
DEADLINE_FRAC = 0.85

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC


def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    if not np.allclose(C, Ctrue, atol=tol):
        return False
    A2 = np.random.default_rng(7).random((n, m))
    B2 = np.random.default_rng(11).random((m, p))
    C2 = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C2, A2 @ B2, atol=tol)


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


def _strassen_2x2_blocks():
    S = [
        ([0, 1], [1, 1]),
        ([1, 0], [1, 1]),
        ([0, 0], [0, 1]),
        ([1, 1], [0, 0]),
        ([1, 0], [0, 0]),
        ([0, 1], [0, 1]),
        ([1, 1], [1, 0]),
    ]
    Cuse = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3],
    ]
    return S, Cuse


def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    S, Cuse = _strassen_2x2_blocks()
    U = np.zeros((nrows * ncolsm, 7), dtype=float)
    V = np.zeros((ncolsm * ncolsk, 7), dtype=float)
    W = np.zeros((ncolsk * nrows, 7), dtype=float)
    for t, (ac, bc) in enumerate(S):
        for i in range(2):
            for j in range(2):
                U[i * ncolsm + j, t] = ac[i][j]
            for k in range(2):
                V[j * ncolsk + k, t] = bc[j][k]
    for i in range(2):
        for k in range(2):
            for t in Cuse[i * 2 + k]:
                W[k * nrows + i, t] = 1.0
    return U, V, W


def _build_445_strassen(n, m, p):
    """4x4x4 Strassen (rank 49) + 16 schoolbook for k=4, total rank 65."""
    S, Cuse = _strassen_2x2_blocks()
    R0 = 7
    U0, V0, W0 = _make_2x2x2_strassen(2, 2, 2)
    U = np.zeros((n * m, 7 * R0), dtype=float)
    V = np.zeros((m * p, 7 * R0), dtype=float)
    W = np.zeros((p * n, 7 * R0), dtype=float)
    for t in range(7):
        ac = S[t][0]; bc = S[t][1]
        for ri in range(2):
            for ci in range(2):
                acoef = ac[ri][ci]
                if acoef == 0:
                    continue
                for ci2 in range(2):
                    for ki in range(2):
                        bcoef = bc[ci2][ki]
                        if bcoef == 0:
                            continue
                        for tp in range(R0):
                            col = t * R0 + tp
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] += acoef * U0[ii * 2 + jj, tp]
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj * p + gk, col] += bcoef * V0[jj2 * 2 + kk, tp]
                        for tp in Cuse[ri * 2 + ki]:
                            col = t * R0 + tp
                            for ii in range(2):
                                for kk in range(2):
                                    gi = ri * 2 + ii
                                    gk = ki * 2 + kk
                                    W[gk * n + gi, col] += W0[kk * 2 + ii, tp]
    # Append schoolbook for k=4
    R = U.shape[1]
    extra = n * m
    Ufull = np.zeros((n * m, R + extra), dtype=float)
    Vfull = np.zeros((m * p, R + extra), dtype=float)
    Wfull = np.zeros((p * n, R + extra), dtype=float)
    Ufull[:, :R] = U
    Vfull[:, :R] = V
    Wfull[:, :R] = W
    k = 4
    c = R
    for i in range(n):
        for j in range(m):
            Ufull[i * m + j, c] = 1.0
            Vfull[j * p + k, c] = 1.0
            Wfull[k * n + i, c] = 1.0
            c += 1
    return Ufull, Vfull, Wfull


def _bini_bordering_search(n, m, p, max_border=12, max_removals=10, rng_seed=42):
    """
    Bini's bordering: start with schoolbook (rank 80), add border of size R_b,
    then try to remove some of the original entries by adjusting border params.
    
    The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    With bordering, we add R_b new columns. The total rank is 80 + R_b.
    We want to find border parameters such that some of the original 80 entries
    become zero (or can be absorbed), effectively reducing the "effective" rank.
    
    Actually, Bini's method works differently: you perturb the tensor and find a
    lower-rank border. Let me implement a direct search instead.
    
    Alternative structural approach: use the fact that 4x4x5 can be seen as
    4x4x4 + 4x4x1, and try to find a rank < 49+16 decomposition.
    
    Let me try a direct optimization: parameterize the decomposition with
    half-integer entries and use random restarts to find low-rank solutions.
    """
    rng = np.random.default_rng(rng_seed)
    best_rank = 80
    best_UVW = None
    
    # Try various rank targets
    for target_rank in range(58, 66):
        if _deadline():
            break
        # Try to find a rank-target_rank decomposition
        # Use random initialization with sparsity
        for attempt in range(3):
            if _deadline():
                break
            U = rng.standard_normal((n * m, target_rank))
            V = rng.standard_normal((m * p, target_rank))
            W = rng.standard_normal((p * n, target_rank))
            
            # Try to optimize: minimize ||T(U,V,W) - T_matmul||
            # This is a non-convex problem, use simple gradient-free approach
            # with random perturbations
            
            best_err = np.inf
            for step in range(200):
                if _deadline():
                    break
                # Compute current tensor error
                Uf = U.reshape(n, m, target_rank)
                Vf = V.reshape(m, p, target_rank)
                Wf = W.reshape(p, n, target_rank)
                C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
                
                # Target: C[i,k] = sum_j i*m+j * j*p+k ... no, target is matmul
                # C_true[i,k] = sum_j A[i,j]*B[j,k] where A[i,j]=i*m+j, B[j,k]=j*p+k
                A = np.arange(n * m, dtype=float).reshape(n, m)
                B = np.arange(m * p, dtype=float).reshape(m, p)
                Ctrue = A @ B
                err = np.sum((C - Ctrue)**2)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-4:
                        break
                
                # Random perturbation
                lr = 0.1 * max(1, np.sqrt(err))
                U += lr * rng.standard_normal((n * m, target_rank))
                V += lr * rng.standard_normal((m * p, target_rank))
                W += lr * rng.standard_normal((p * n, target_rank))
            
            if best_err < 1e-4:
                best_rank = target_rank
                best_UVW = (U.copy(), V.copy(), W.copy())
                break
    
    if best_UVW is not None:
        return best_UVW
    return None


def construct():
    n, m, p = DIMENSIONS
    best = None

    # Candidate 1: Strassen composition (rank 65)
    if not _deadline():
        try:
            U2, V2, W2 = _build_445_strassen(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass

    # Candidate 2: Bini bordering search
    if not _deadline():
        try:
            result = _bini_bordering_search(n, m, p, max_border=12, max_removals=10, rng_seed=42)
            if result is not None:
                U3, V3, W3 = result
                if _verify(U3, V3, W3, n, m, p):
                    if best is None or U3.shape[1] < best[3]:
                        best = (U3, V3, W3, U3.shape[1])
        except Exception:
            pass

    # Candidate 3: schoolbook (rank 80)
    try:
        Us, Vs, Ws = _schoolbook(n, m, p)
        if _verify(Us, Vs, Ws, n, m, p):
            if best is None or Us.shape[1] < best[3]:
                best = (Us, Vs, Ws, Us.shape[1])
    except Exception:
        pass

    if best is None:
        Us, Vs, Ws = _schoolbook(n, m, p)
        best = (Us, Vs, Ws, Us.shape[1])

    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
