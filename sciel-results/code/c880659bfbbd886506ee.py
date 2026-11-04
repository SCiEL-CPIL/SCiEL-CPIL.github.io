import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def time_left():
        return deadline - time.time()

    # ---- Build a Strassen-based rank-42 decomposition (exact integers) ----
    # Tiled 2x2x2 Strassen blocks: i in {0}, j in {0,2}, k in {0,2,4} -> 6 blocks * 7 = 42
    RANK_INIT = 42
    U = np.zeros((nm, RANK_INIT), dtype=float)
    V = np.zeros((mp, RANK_INIT), dtype=float)
    W = np.zeros((pn, RANK_INIT), dtype=float)

    def add_strassen_2x2x2(i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1

        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1

        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1

        # M1 = (A00+A11)(B00+B11)
        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0

        # M2 = (A10+A11)B00
        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0

        # M3 = A00(B01-B11)
        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0

        # M4 = A11(B10-B00)
        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0

        # M5 = (A00+A01)B11
        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0

        # M6 = (A10-A00)(B00+B01)
        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0

        # M7 = (A01-A11)(B10+B11)
        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_2x2x2(i_start, j_start, k_start, offset)
                offset += 7

    U = U[:, :offset]
    V = V[:, :offset]
    W = W[:, :offset]

    # ---- Exact verification (Fraction) ----
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]; must equal 1 iff (a,b,c) is a
    # valid (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    def to_frac(M):
        return [[Fraction(int(round(M[i, j]))) for j in range(M.shape[1])]
                for i in range(M.shape[0])]

    def verify_exact(Um, Vm, Wm):
        r = Um.shape[1]
        Uf, Vf, Wf = to_frac(Um), to_frac(Vm), to_frac(Wm)
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i * m + j, j * p + k, k * n + i))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = Fraction(0)
                    for r_idx in range(r):
                        s += Uf[a][r_idx] * Vf[b][r_idx] * Wf[c][r_idx]
                    want = Fraction(1) if (a, b, c) in valid else Fraction(0)
                    if s != want:
                        return False
        return True

    # ---- Fast float verification (cheap, for search) ----
    def verify_fast(Um, Vm, Wm, tol=1e-7):
        r = Um.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = float(np.dot(Um[a], Vm[b] * Wm[c]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # ---- Sanity: verify the base decomposition ----
    if not verify_fast(U, V, W):
        # Fallback to schoolbook rank n*m*p
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]

    # ---- Refinement: iterative column removal (rank reduction) ----
    # Try to drop a column while the tensor is still exactly reconstructed.
    # Each successful removal strictly reduces rank; repeat until none work.
    def try_remove(Um, Vm, Wm):
        r = Um.shape[1]
        for col in range(r - 1, -1, -1):
            if time.time() > deadline - 8:
                return None
            U2 = np.delete(Um, col, axis=1)
            V2 = np.delete(Vm, col, axis=1)
            W2 = np.delete(Wm, col, axis=1)
            if verify_fast(U2, V2, W2):
                return (U2, V2, W2)
        return None

    # Greedy removal loop
    while time.time() < deadline - 8:
        res = try_remove(best_U, best_V, best_W)
        if res is None:
            break
        best_U, best_V, best_W = res
        best_rank = best_U.shape[1]

    # ---- Final exact polish: confirm exactness of the reduced decomposition ----
    if verify_exact(best_U, best_V, best_W):
        final_U, final_V, final_W = best_U, best_V, best_W
    else:
        # Fall back to the verified base decomposition
        final_U, final_V, final_W = U, V, W

    return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
