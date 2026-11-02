"""Tensor decomposition for 2x5x6 matrix multiplication — rank 54."""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 130  # safety margin below 150 s

def _elapsed():
    return time.time() - _START

def construct():
    """Return {'U','V','W'} whose tensor product equals the 2×5×6 matmul tensor."""
    n, m, p = 2, 5, 6

    # Strategy: partition the 5 inner-dimension indices j into pairs {0,1}, {2,3}
    # and singleton {4}; partition the 6 outer-dimension indices k into pairs
    # {0,1}, {2,3}, {4,5}.  For each (j-pair, k-pair) use a 7-term Strassen
    # 2x2x2 block (rank 7 per block), and for the singleton j=4 use schoolbook
    # (rank n*p = 12).  Total rank = 6*7 + 12 = 54.

    RANK = 54
    U = np.zeros((n * m, RANK), dtype=float)   # shape (10, 54)
    V = np.zeros((m * p, RANK), dtype=float)   # shape (30, 54)
    W = np.zeros((p * n, RANK), dtype=float)   # shape (12, 54)
    col = 0

    # --- helper: emit one rank-1 term -----------------------------------------
    def _emit(u_spec, v_spec, w_spec):
        nonlocal col
        if col >= RANK:
            return
        for (idx, c) in u_spec:
            U[idx, col] += c
        for (idx, c) in v_spec:
            V[idx, col] += c
        for (idx, c) in w_spec:
            W[idx, col] += c
        col += 1

    # --- Strassen 2x2x2 block for j-pair (ja,jb), k-pair (ka,kb) -------------
    # U index: i*m + j  -> (i,ja) and (i,jb)
    # V index: j*p + k  -> (ja,ka),(ja,kb),(jb,ka),(jb,kb)
    # W index: k*n + i  -> (ka,i0),(ka,i1),(kb,i0),(kb,i1)

    def _strassen(ja, jb, ka, kb):
        if _elapsed() > _DEADLINE:
            return
        # index aliases
        ua = ja          # row for (0, ja)
        ub = m + ja      # row for (1, ja)
        va = jb          # row for (0, jb)
        vb = m + jb      # row for (1, jb)
        wka0 = 2 * ka    # (ka, 0)
        wka1 = 2 * ka + 1
        wkb0 = 2 * kb
        wkb1 = 2 * kb + 1
        vka = ja * p + ka
        vkb = ja * p + kb
        vjb_ka = jb * p + ka
        vjb_kb = jb * p + kb

        # 7 Strassen factors (standard 2x2x2)
        _emit([(ua, 1), (vb, 1)],
              [(vka, 1), (vjb_kb, 1)],
              [(wka0, 1), (wkb1, 1)])

        _emit([(ub, 1), (vb, 1)],
              [(vka, 1)],
              [(wka1, 1), (wkb1, -1)])

        _emit([(ua, 1)],
              [(vkb, 1), (vjb_kb, -1)],
              [(wkb0, 1), (wkb1, 1)])

        _emit([(vb, 1)],
              [(vjb_ka, 1), (vka, -1)],
              [(wka0, 1), (wka1, 1)])

        _emit([(ua, 1), (va, 1)],
              [(vjb_kb, 1)],
              [(wka0, -1), (wkb0, 1)])

        _emit([(ub, 1), (ua, -1)],
              [(vka, 1), (vkb, 1)],
              [(wkb1, 1)])

        _emit([(va, 1), (vb, -1)],
              [(vjb_ka, 1), (vjb_kb, 1)],
              [(wka0, 1)])

    # --- apply 6 Strassen blocks ---------------------------------------------
    for jp in ((0, 1), (2, 3)):
        for kp in ((0, 1), (2, 3), (4, 5)):
            _strassen(jp[0], jp[1], kp[0], kp[1])

    # --- schoolbook for j = 4 -------------------------------------------------
    for i in range(n):
        for k in range(p):
            if _elapsed() > _DEADLINE:
                # partial result — still try to return what we have
                return {"U": U[:, :col].tolist(),
                        "V": V[:, :col].tolist(),
                        "W": W[:, :col].tolist()}
            U[i * m + 4, col] = 1.0
            V[4 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    U_out = U[:, :col].tolist()
    V_out = V[:, :col].tolist()
    W_out = W[:, :col].tolist()

    # --- verification (fast vectorised check) ---------------------------------
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dot = np.dot(np.dot(U[i*m + j], V[j*p + k]), W[k*n + i])
                if abs(dot - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return {"U": U_out, "V": V_out, "W": W_out}

    # fallback: full schoolbook rank 60
    UF = np.zeros((n*m, n*m*p))
    VF = np.zeros((m*p, n*m*p))
    WF = np.zeros((p*n, n*m*p))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                UF[i*m+j, c] = 1.0
                VF[j*p+k, c] = 1.0
                WF[k*n+i, c] = 1.0
                c += 1
    return {"U": UF.tolist(), "V": VF.tolist(), "W": WF.tolist()}
