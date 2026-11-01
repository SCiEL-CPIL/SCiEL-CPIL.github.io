"""Rewritten solver: fast vectorized ALS + exact rational verification for 2x2 x 2x5 matmul."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N
SCHOOLBOOK = N * M * P  # 20

_T0 = time.time()
DEADLINE = 150.0
SAFETY = 6.0

# ---------------------------------------------------------------------------
# Build the multiplication tensor T with indices (i, jp, k) -> scalar
# T[i,jp,k] = 1 if j == jp (where j indexes U's column), else 0.
# Decomposition: T[i,jp,k] = sum_c U[i,j,c] * V[jp,k,c] * W[k,i,c]
# ---------------------------------------------------------------------------
def _build_T():
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for jp in range(M):
                for k in range(P):
                    if j == jp:
                        T[i, jp, k] = 1.0
    return T

T = _build_T()

# Precompute Kronecker structure for ALS.
# For a fixed c, the scalar contribution to T[i,jp,k] is U[i,j,c]*V[jp,k,c]*W[k,i,c].
# Treat U as a (N*M) vector (row i*m+j), V as (M*P) vector (row jp*p+k), W as (P*N) (row k*n+i).
# Then sum_c u_c (i,j) * v_c (jp,k) * w_c (k,i) = T(i,jp,k).
#
# ALS for U (holding V,W): for each (i,jp,k), coefficient on U[i*m+j] is v[jp*p+k]*w[k*n+i].
# We need: sum over (jp,k) of v[jp*p+k]*w[k*n+i] * U[i*m+j] ... but U[i*m+j] is shared across jp.
# Rearrange: for fixed i, define for each j:  sum_{jp,k} T[i,jp,k] = sum_j U[i*m+j] * (sum_{jp,k} v[jp*p+k]*w[k*n+i]*delta_{j,jp})
# => sum_{k} v[j*p+k]*w[k*n+i] * U[i*m+j]  (since jp==j)
# So the "linear system" for U[i*m+j] (j fixed, i fixed):
#   sum_k [ v[j*p+k] * w[k*n+i] ] * U[i*m+j] = T[i, j, k] summed over k? No, T depends on k.
# This is a 3-way tensor; ALS solves by flattening the other two modes.
# Standard ALS: fix V,W, solve for U. The objective is sum_{i,jp,k} (T[i,jp,k] - sum_c U[i*m+j,c] V[jp*p+k,c] W[k*n+i,c])^2.
# For U, this is a linear least squares. Vectorize:
#   Let A be a matrix where each row corresponds to (i,jp,k). The column for variable U[i*m+j,c] is V[jp*p+k,c]*W[k*n+i,c].
# Build once per (c) using the structure.

def _build_A_U():
    # rows = (i, jp, k) flattened; cols = (i, j, c) flattened
    rows = []
    for i in range(N):
        for jp in range(M):
            for k in range(P):
                rows.append((i, jp, k))
    n_rows = len(rows)
    A = np.zeros((n_rows, NM * 1))  # we build per c, so cols = NM
    # For a given c, col index (i*m+j) gets value V[jp*p+k,c]*W[k*n+i,c] only when j == jp, else 0.
    return rows

_ROWS_U = [(i, jp, k) for i in range(N) for jp in range(M) for k in range(P)]
_bT = T.reshape(-1)  # index (i, jp, k)

def _als_solve_U(V, W, c):
    # V shape (MP, rank), W shape (PN, rank); solve for U[:, c]
    rhs = np.zeros(NM)
    for idx, (i, jp, k) in enumerate(_ROWS_U):
        val = V[jp * P + k, c] * W[k * N + i, c]
        if val != 0.0:
            rhs[i * M + jp] += val * _bT[idx]
    return rhs

def _als_solve_V(U, W, c):
    rhs = np.zeros(MP)
    for idx, (i, jp, k) in enumerate(_ROWS_U):
        val = U[i * M + jp, c] * W[k * N + i, c]
        if val != 0.0:
            rhs[jp * P + k] += val * _bT[idx]
    return rhs

def _als_solve_W(U, V, c):
    rhs = np.zeros(PN)
    for idx, (i, jp, k) in enumerate(_ROWS_U):
        val = U[i * M + jp, c] * V[jp * P + k, c]
        if val != 0.0:
            rhs[k * N + i] += val * _bT[idx]
    return rhs

def _als_step(U, V, W, rank):
    for c in range(rank):
        U[:, c] = _als_solve_U(V, W, c)
    for c in range(rank):
        V[:, c] = _als_solve_V(U, W, c)
    for c in range(rank):
        W[:, c] = _als_solve_W(U, V, c)

def _objective(U, V, W, rank):
    # sum over (i,jp,k) of (T - sum_c U V W)^2
    err = 0.0
    for idx, (i, jp, k) in enumerate(_ROWS_U):
        s = 0.0
        for c in range(rank):
            s += U[i * M + jp, c] * V[jp * P + k, c] * W[k * N + i, c]
        d = s - _bT[idx]
        err += d * d
    return err

# ---------------------------------------------------------------------------
# Exact verification using Fraction
# ---------------------------------------------------------------------------
def _snap_half(x):
    return Fraction(round(x * 2), 2)

def _verify_exact(U, V, W, rank):
    Uf = [[_snap_half(U[i, c]) for c in range(rank)] for i in range(NM)]
    Vf = [[_snap_half(V[i, c]) for c in range(rank)] for i in range(MP)]
    Wf = [[_snap_half(W[i, c]) for c in range(rank)] for i in range(PN)]
    for i in range(N):
        for jp in range(M):
            for k in range(P):
                target = 1 if (i * M + jp) and False else (1 if True else 0)
                # target = 1 iff j == jp where j is the U column index = (i*m+j); but here jp is the V row's j.
                # In T, the index is (i, jp, k) and T=1 iff the U-column j equals jp.
                # Since U column is (i*m+j) with j==jp, target = 1 always? No.
                # T[i,jp,k] = 1 if j == jp. But j is a free index in U. The decomposition sums over c,
                # and U[i*m+j,c] is the only U entry with column j. So effectively for a fixed (i,jp,k),
                # the U entry used is U[i*m+jp,c] (j=jp). Hence target = 1 for all (i,jp,k)?? 
                # Let's re-examine: T[i,jp,k] = delta_{j,jp} where j is the column of U = the "input" index.
                # Actually the matrix product (AB)[i,k] = sum_j A[i,j] B[j,k].
                # U encodes A[i,j], V encodes B[j,k], W encodes the output routing to (i,k).
                # The tensor T[i,jp,k] = delta_{j, jp} means: the product term A[i,j]*B[j,k] contributes to output (i,k).
                # So for the decomposition to reproduce (AB)[i,k] = sum_j A[i,j]B[j,k], we need
                #   sum_c U[i*m+j,c] V[jp*p+k,c] W[k*n+i,c] = delta_{j,jp}
                # for ALL j, jp, i, k. So target = 1 if j==jp else 0, and j is an independent index.
                # In my _build_T I indexed by (i, jp, k) only, losing j. I must include j.
                # Fix: T is 4-indexed (i, j, jp, k) = delta_{j,jp}. Let me rebuild properly.
                pass
    return False

# I made an indexing error: T must be 4-indexed (i, j, jp, k) = delta_{j,jp}.
# Rebuild T with the correct 4 indices.
def _build_T4():
    T4 = {}
    for i in range(N):
        for j in range(M):
            for jp in range(M):
                for k in range(P):
                    T4[(i, j, jp, k)] = 1.0 if j == jp else 0.0
    return T4

T4 = _build_T4()
_KEYS = list(T4.keys())
_T4vals = np.array([T4[key] for key in _KEYS], dtype=float)

def _als_solve_U4(V, W, c):
    # variable U[i*m+j, c]; for each (i,j,jp,k) with target, coefficient V[jp*p+k,c]*W[k*n+i,c]
    rhs = np.zeros(NM)
    for key, tval in zip(_KEYS, _T4vals):
        i, j, jp, k = key
        val = V[jp * P + k, c] * W[k * N + i, c]
        if val != 0.0:
            rhs[i * M + j] += val * tval
    return rhs

def _als_solve_V4(U, W, c):
    rhs = np.zeros(MP)
    for key, tval in zip(_KEYS, _T4vals):
        i, j, jp, k = key
        val = U[i * M + j, c] * W[k * N + i, c]
        if val != 0.0:
            rhs[jp * P + k] += val * tval
    return rhs

def _als_solve_W4(U, V, c):
    rhs = np.zeros(PN)
    for key, tval in zip(_KEYS, _T4vals):
        i, j, jp, k = key
        val = U[i * M + j, c] * V[jp * P + k, c]
        if val != 0.0:
            rhs[k * N + i] += val * tval
    return rhs

def _als_step4(U, V, W, rank):
    for c in range(rank):
        U[:, c] = _als_solve_U4(V, W, c)
    for c in range(rank):
        V[:, c] = _als_solve_V4(U, W, c)
    for c in range(rank):
        W[:, c] = _als_solve_W4(U, V, c)

def _objective4(U, V, W, rank):
    err = 0.0
    for key, tval in zip(_KEYS, _T4vals):
        i, j, jp, k = key
        s = 0.0
        for c in range(rank):
            s += U[i * M + j, c] * V[jp * P + k, c] * W[k * N + i, c]
        d = s - tval
        err += d * d
    return err

def _verify_exact4(U, V, W, rank):
    Uf = [[_snap_half(U[i, c]) for c in range(rank)] for i in range(NM)]
    Vf = [[_snap_half(V[i, c]) for c in range(rank)] for i in range(MP)]
    Wf = [[_snap_half(W[i, c]) for c in range(rank)] for i in range(PN)]
    for key in _KEYS:
        i, j, jp, k = key
        target = Fraction(1) if j == jp else Fraction(0)
        s = Fraction(0)
        for c in range(rank):
            s += Uf[i * M + j][c] * Vf[jp * P + k][c] * Wf[k * N + i][c]
        if s != target:
            return False
    return True

# ---------------------------------------------------------------------------
# ALS driver with many restarts
# ---------------------------------------------------------------------------
def _als_try(rank, n_restarts, tol=1e-9, max_inner=60, seed_base=0):
    best = None
    for r in range(n_restarts):
        if time.time() > DEADLINE - SAFETY:
            break
        np.random.seed(seed_base + r * 7919)
        U = np.random.uniform(-1, 1, (NM, rank))
        V = np.random.uniform(-1, 1, (MP, rank))
        W = np.random.uniform(-1, 1, (PN, rank))
        prev = _objective4(U, V, W, rank)
        for it in range(max_inner):
            if time.time() > DEADLINE - SAFETY:
                break
            _als_step4(U, V, W, rank)
            cur = _objective4(U, V, W, rank)
            if abs(prev - cur) < tol:
                prev = cur
                break
            prev = cur
        # cheap exact check on the (possibly) converged solution
        if _verify_exact4(U, V, W, rank):
            return U, V, W
        # keep best-so-far by objective for a final polish
        if best is None or cur < best[0]:
            best = (cur, U.copy(), V.copy(), W.copy())
    if best is not None:
        _, U, V, W = best
        # a couple more polish steps
        for _ in range(30):
            if time.time() > DEADLINE - SAFETY:
                break
            _als_step4(U, V, W, rank)
        if _verify_exact4(U, V, W, rank):
            return U, V, W
    return None

def _structured_init(rank):
    """Strassen-style structured initial guess for small ranks."""
    U = np.zeros((NM, rank)); V = np.zeros((MP, rank)); W = np.zeros((PN, rank))
    # A few known useful tensors: place identity-like patterns
    for c in range(rank):
        # spread some 1s
        U[np.random.randint(NM), c] = 1.0
        V[np.random.randint(MP), c] = 1.0
        W[np.random.randint(PN), c] = 1.0
    return U, V, W

def _schoolbook():
    rank = SCHOOLBOOK
    U = np.zeros((NM, rank)); V = np.zeros((MP, rank)); W = np.zeros((PN, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def construct():
    # Try ranks from a promising lower bound up to schoolbook.
    # Theoretical lower bound ~ 13-14; try 13,14,15,16,17,18 then fallback.
    for rank in [13, 14, 15, 16, 17, 18, 19]:
        if time.time() > DEADLINE - SAFETY:
            break
        res = _als_try(rank, n_restarts=40, seed_base=rank * 101)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # fallback schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
