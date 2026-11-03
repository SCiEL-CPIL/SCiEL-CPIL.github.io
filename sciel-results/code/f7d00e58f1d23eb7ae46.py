"""Matrix multiplication tensor decomposition for 4x4 * 4x5.
Core: Strassen (rank 49) for the 4x4*4x4 part + schoolbook (rank 16) for the extra
column -> rank 65. Then refine: (a) fast linear-dependency rank reduction, (b) bounded
random search for lower-rank integer/half-integer decompositions.
Guaranteed-valid rank-65 result is always the floor.
"""
import time
import random
import numpy as np

# ---- Fast tensor machinery (precomputed index maps) -------------------------
_n, _m, _p = 4, 4, 5
_N, _M, _P = _n * _m, _m * _p, _p * _n  # 16, 20, 20

# _IU[a, b, c] = index of the rank-1 term (a,b,c) in the schoolbook basis.
_IU = {}
_IU_LIST = []
for _i in range(_n):
    for _j in range(_m):
        for _k in range(_p):
            _IU[(i, j, k)] = len(_IU_LIST)
            _IU_LIST.append((i * _m + j, j * _p + k, k * _n + i))
# _Uidx[a,b,c] -> row in U ; _Vidx -> row in V ; _Widx -> row in W
_Uidx = np.array([t[0] for t in _IU_LIST], dtype=np.int64)
_Vidx = np.array([t[1] for t in _IU_LIST], dtype=np.int64)
_Widx = np.array([t[2] for t in _IU_LIST], dtype=np.int64)


def _tensor(U, V, W):
    """Compute the full multiplication tensor from factors (float)."""
    return U[_Uidx] * V[_Vidx] * W[_Widx]


def _is_valid(U, V, W, tol=1e-9):
    """Check T[i,j,k] == 1 for all i,j,k."""
    t = _tensor(U, V, W)
    return bool(np.all(np.abs(t - 1.0) <= tol))


def _verify(U, V, W):
    """Same as _is_valid (kept for interface parity)."""
    return _is_valid(U, V, W)


# ---- Strassen 2x2 -----------------------------------------------------------
def _strassen_2x2():
    rank = 7
    U = np.zeros((4, rank), dtype=np.int64)
    V = np.zeros((4, rank), dtype=np.int64)
    W = np.zeros((4, rank), dtype=np.int64)
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    return U, V, W


# ---- Strassen 4x4 * 4x4 (rank 49) ------------------------------------------
def _build_strassen_4x4_4x4():
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    sU, sV, sW = _strassen_2x2()
    mults = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 0: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: 1}),
    ]
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        for s in range(7):
            col = mi * 7 + s
            for xi in range(4):
                u_val = sU[xi, s]
                if u_val == 0:
                    continue
                xr = xi // 2
                xc = xi % 2
                for ab, coeff in a_c.items():
                    U[ab * 4 + xr * 2 + xc, col] += u_val * coeff
            for yi in range(4):
                v_val = sV[yi, s]
                if v_val == 0:
                    continue
                yr = yi // 2
                yc = yi % 2
                for bb, coeff in b_c.items():
                    V[bb * 4 + yr * 2 + yc, col] += v_val * coeff
            for zi in range(4):
                w_val = sW[zi, s]
                if w_val == 0:
                    continue
                zr = zi // 2
                zc = zi % 2
                for cb, coeff in c_c.items():
                    W[cb * 4 + zr * 2 + zc, col] += w_val * coeff
    return U, V, W


# ---- Extra column (4x4 * 4x1, rank 16) -------------------------------------
def _build_extra_column():
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _schoolbook():
    """Full schoolbook decomposition (rank 80)."""
    n, m, p = _n, _m, _p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_base():
    """Rank-65 base: Strassen 4x4x4 (cols 0-3) + schoolbook extra column (col 4)."""
    n, m, p = _n, _m, _p
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U_e, V_e, W_e = _build_extra_column()
    total_rank = 49 + 16
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    U[:, :49] = U_s
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V_s[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W_s[k * 4 + i, :]
    U[:, 49:] = U_e
    for j in range(m):
        V[j * p + 4, 49:] = V_e[j * 1 + 0, :]
    for i in range(n):
        W[4 * n + i, 49:] = W_e[0 * n + i, :]
    return U, V, W


# ---- Fast rank reduction (linear dependence via vectorized tensor test) -----
def _try_rank_reduction(U, V, W, deadline):
    """Greedy: remove columns that leave the decomposition valid. Fast vectorized test."""
    rank = U.shape[1]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        r = U.shape[1]
        for c in range(r):
            if time.time() > deadline:
                break
            U_t = np.delete(U, c, axis=1)
            V_t = np.delete(V, c, axis=1)
            W_t = np.delete(W, c, axis=1)
            t = _tensor(U_t, V_t, W_t)
            if np.all(np.abs(t - 1.0) <= 1e-9):
                U, V, W = U_t, V_t, W_t
                changed = True
                break  # restart scan with new rank
    return U, V, W


# ---- Random search for lower-rank decompositions ----------------------------
def _random_search(U, V, W, best_rank, deadline):
    """Bounded random search over integer/half-integer entries for rank < best_rank."""
    best_rank_local = best_rank
    best_sol = None
    # Ranks to try, low to high, but always below the current best.
    # Theoretical lower bound for 4x4*4x5 is >= max(n,p)=5, practical ~ 20-30.
    # We only spend time here if it cannot exceed the current best.
    candidates = []
    for r in range(16, best_rank_local):
        candidates.append(r)
    # Fewer trials at higher ranks; more at lower.
    for r in candidates:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        # ~ each trial is a vectorized tensor build (fast). Cap trials by time.
        num_trials = max(8, min(400, int(remaining * 40)))
        # Lower ranks -> slightly more trials, capped.
        if r <= 24:
            num_trials = min(num_trials + 100, 600)
        choices_int = [-1, 0, 1]
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            # Use integer entries for the bulk; half-integers occasionally.
            if (trial % 5 == 4):
                choices = [-1.0, -0.5, 0.0, 0.5, 1.0]
            else:
                choices = choices_int
            U_r = np.array([[random.choice(choices) for _ in range(r)] for _ in range(_N)],
                           dtype=np.float64)
            V_r = np.array([[random.choice(choices) for _ in range(r)] for _ in range(_M)],
                           dtype=np.float64)
            W_r = np.array([[random.choice(choices) for _ in range(r)] for _ in range(_P)],
                           dtype=np.float64)
            t = _tensor(U_r, V_r, W_r)
            if np.all(np.abs(t - 1.0) <= 1e-9):
                best_rank_local = r
                # Convert to a clean representation (int if possible).
                best_sol = (U_r, V_r, W_r)
                break  # found at this rank; try lower next
        if best_sol is not None:
            # Found something; keep searching lower ranks.
            continue
    return best_rank_local, best_sol


def construct():
    n, m, p = _n, _m, _p
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s wall-clock limit

    # 1) Build guaranteed-valid rank-65 base.
    U, V, W = _build_base()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Verify base; if it fails (shouldn't), fall back to schoolbook.
    if not _verify(U, V, W):
        U, V, W = _schoolbook()
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W

    # 2) Fast linear-dependency rank reduction on the base.
    if time.time() < deadline - 20:
        U, V, W = _try_rank_reduction(U, V, W, deadline - 20)
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U, V, W

    # 3) Bounded random search for genuinely lower-rank decompositions.
    if time.time() < deadline - 5 and best_rank > 20:
        new_rank, new_sol = _random_search(best_U, best_V, best_W, best_rank, deadline - 5)
        if new_sol is not None and new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = new_sol

    # Final verification of the best solution; if invalid, use schoolbook.
    if not _verify(best_U, best_V, best_W):
        best_U, best_V, best_W = _schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
