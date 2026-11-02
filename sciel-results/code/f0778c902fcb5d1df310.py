"""Matrix multiplication 4,4,9: rank-reduction search from schoolbook."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
N, M, P = DIMENSIONS
NM = N * M
MP = M * P
PN = P * N
RANK0 = N * M * P          # 144 schoolbook rank

def _time_left(deadline):
    return deadline - time.time()

def construct():
    start = time.time()
    deadline = start + 140.0          # safety margin vs 150s wall-clock
    rng = random.Random(20240517)

    # ---- Build schoolbook term list (rank-1 tensors) ----
    # U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1  for each (i,j,k)
    Us = []
    Vs = []
    Ws = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = np.zeros(NM)
                v = np.zeros(MP)
                w = np.zeros(PN)
                u[i * M + j] = 1.0
                v[j * P + k] = 1.0
                w[k * N + i] = 1.0
                Us.append(u); Vs.append(v); Ws.append(w)
    R0 = len(Us)
    A = np.array(Us).T          # (R0, NM)
    B = np.array(Vs).T          # (R0, MP)
    C = np.array(Ws).T          # (R0, PN)

    def tensor(r):
        return A[r] @ np.diag(B[r]) @ C[r]

    def is_zero(r):
        t = tensor(r)
        return not (np.abs(t) > 1e-9).any()

    def is_zero_fast(r):
        # random linear probe of the tensor (cheap necessary check)
        a = rng.gauss(0, 1)
        b = rng.gauss(0, 1)
        c = rng.gauss(0, 1)
        return abs(a * B[r] * C[r]) < 1e-12  # structural: single 1 in each -> not zero unless coeff 0
    # (is_zero_fast not actually used; full check is cheap enough)

    # ---- Candidate state: U,V,W as (rows, R0) matrices ----
    U = np.eye(R0, dtype=float)
    V = np.eye(R0, dtype=float)
    W = np.eye(R0, dtype=float)
    active = list(range(R0))
    best_rank = R0
    best = (U.copy(), V.copy(), W.copy(), list(active))

    def snapshot():
        nonlocal best_rank, best
        if len(active) < best_rank:
            best_rank = len(active)
            best = (U.copy(), V.copy(), W.copy(), list(active))

    def full_check():
        """Return True if current active set is an exact decomposition."""
        for r in active:
            if not is_zero(r):
                return False
        return True

    def try_move(Uc, Vc, Wc, activec):
        """If all activec are zero, commit and return True."""
        for r in activec:
            if not is_zero(r):
                return False
        nonlocal U, V, W, active
        U, V, W = Uc, Vc, Wc
        active = list(activec)
        snapshot()
        return True

    # ---- Phase 1: random linear combos to find zero terms (rank reduction) ----
    # Represent U,V,W as linear maps on the R0 coefficient space.
    # A move: pick a target column r and a set S of other columns, set
    #   new_col_r = col_r + sum_{s in S} c_s * col_s
    # for U, V, W independently (same coefficients per matrix? no, independent).
    # This is a rank-preserving change of basis among active columns.

    def random_basis_change():
        if len(active) < 2:
            return False
        r = rng.choice(active)
        others = [x for x in active if x != r]
        if not others:
            return False
        # small random subset
        sz = rng.randint(1, min(3, len(others)))
        S = rng.sample(others, sz)
        Uc = U.copy(); Vc = V.copy(); Wc = W.copy()
        for mat, base in ((Uc, U), (Vc, V), (Wc, W)):
            for s in S:
                mat[:, r] += rng.gauss(0, 1) * base[:, s]
        return try_move(Uc, Vc, Wc, active)

    # ---- Phase 2: random rank-reduction by expressing one active col as combo ----
    def random_reduce():
        if len(active) < 3:
            return False
        r = rng.choice(active)
        others = [x for x in active if x != r]
        sz = rng.randint(1, min(2, len(others)))
        S = rng.sample(others, sz)
        Uc = U.copy(); Vc = V.copy(); Wc = W.copy()
        for mat, base in ((Uc, U), (Vc, V), (Wc, W)):
            mat[:, r] = sum(rng.gauss(0, 1) * base[:, s] for s in S)
        new_active = [x for x in active if x != r]
        return try_move(Uc, Vc, Wc, new_active)

    iters = 0
    while _time_left(deadline) > 1.0:
        iters += 1
        # bias toward reduction attempts
        if rng.random() < 0.5 and len(active) >= 3:
            random_reduce()
        else:
            random_basis_change()

    # ---- Final: return best valid decomposition ----
    Uf, Vf, Wf, act = best
    Uout = Uf[:, act].tolist()
    Vout = Vf[:, act].tolist()
    Wout = Wf[:, act].tolist()
    return {"U": Uout, "V": Vout, "W": Wout}
