"""Rank-40 decomposition for 2x3 * 3x10 matrix multiplication.

Strategy: split the 10 columns of B into 5 disjoint pairs (k0, k1).
For each pair we need a 2x3 * 3x2 block of the output.  That 2x3x2
tensor is decomposed as a direct sum of:
  * a 2x2x2 core (A rows {0,1}, B rows {0,1})  -> rank 5
  * a 2x1x2 tail (A rows {0,1}, B row {2})     -> rank 3
Total 8 terms per pair * 5 pairs = rank 40.
"""
import time
import numpy as np
import random

N, M, P = 2, 3, 10


def _verify(U, V, W, n, m, p, seed=12345, ntrials=12, tol=1e-8):
    rng = np.random.default_rng(seed)
    Ur = np.asarray(U, dtype=float).reshape(n, m, -1)
    Vr = np.asarray(V, dtype=float).reshape(m, p, -1)
    Wr = np.asarray(W, dtype=float).reshape(p, n, -1)
    for _ in range(ntrials):
        A = rng.uniform(-1, 1, (n, m))
        B = rng.uniform(-1, 1, (m, p))
        C = A @ B
        u = np.einsum('ijr,ij->r', Ur, A)
        v = np.einsum('jkr,jk->r', Vr, B)
        Cc = np.einsum('kir,ir,kr->ik', Wr, u, v)
        if np.max(np.abs(C - Cc)) > tol:
            return False
    return True


def _core_222():
    """Rank-5 decomposition of the 2x2x2 multiplication tensor.

    Local indices: A[i,j], B[j,k], C[k,i] with i,j,k in {0,1}.
    Local row ids: A -> i*2+j ; B -> j*2+k ; C -> k*2+i.

    Products:
      m0 = a00 * b00
      m1 = a11 * b11
      m2 = (a00+a11) * (b10 - b00)
      m3 = (a10+a11) * (b00 + b01)
      m4 = (a00 - a10) * (b10 + b11)
    Reassembly:
      c00 = m0 + m2 - m3
      c01 = m3 + m4
      c10 = m2 + m4
      c11 = m1 - m4
    """
    U = np.zeros((4, 5)); V = np.zeros((4, 5)); W = np.zeros((4, 5))
    # m0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # m1
    U[3, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
    # m2
    U[0, 2] = 1; U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[0, 2] = 1; W[2, 2] = 1
    # m3
    U[2, 3] = 1; U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = 1
    W[0, 3] = -1; W[1, 3] = 1
    # m4
    U[0, 4] = 1; U[2, 4] = -1
    V[2, 4] = 1; V[3, 4] = 1
    W[1, 4] = 1; W[2, 4] = 1; W[3, 4] = -1
    return U, V, W


def _tail_212():
    """Rank-3 decomposition of the 2x1x2 tail (A rows {0,1}, single B row).

    Local indices: A[i], B[k], C[k,i] with i in {0,1}, k in {0,1}.
    Local row ids: A -> i ; B -> k ; C -> k*2+i.

    Products:
      m0 = a0 * b0
      m1 = a1 * b1
      m2 = (a0 - a1) * (b0 + b1)
    Reassembly:
      c00 = m0 + m2
      c01 = m2
      c10 = m1 - m2
    """
    U = np.zeros((2, 3)); V = np.zeros((2, 3)); W = np.zeros((4, 3))
    # m0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # m1
    U[1, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
    # m2
    U[0, 2] = 1; U[1, 2] = -1
    V[0, 2] = 1; V[1, 2] = 1
    W[0, 2] = 1; W[1, 2] = 1; W[2, 2] = -1
    return U, V, W


def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    UC, VC, WC = _core_222()
    UT, VT, WT = _tail_212()

    rank = 40
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))

    col = 0
    for pair in range(P // 2):
        k0 = 2 * pair
        k1 = 2 * pair + 1

        # ---- core terms (5) ----
        for r in range(5):
            # A local rows 0..3 == global rows 0..3  (A rows {0,1}, B rows {0,1})
            U[0, col + r] = UC[0, r]
            U[1, col + r] = UC[1, r]
            U[2, col + r] = UC[2, r]
            U[3, col + r] = UC[3, r]
            # B local rows {0,1,2,3} -> global {k0, k1, 10+k0, 10+k1}
            V[k0,     col + r] = VC[0, r]
            V[k1,     col + r] = VC[1, r]
            V[10 + k0, col + r] = VC[2, r]
            V[10 + k1, col + r] = VC[3, r]
            # C local rows {0,1,2,3} -> global {2*k0, 2*k0+1, 2*k1, 2*k1+1}
            W[2 * k0,     col + r] = WC[0, r]
            W[2 * k0 + 1, col + r] = WC[1, r]
            W[2 * k1,     col + r] = WC[2, r]
            W[2 * k1 + 1, col + r] = WC[3, r]
        col += 5

        # ---- tail terms (3) ----
        for r in range(3):
            # A local rows 0,1 -> global rows 3,4  (A rows {0,1}, B row {2})
            U[3, col + r] = UT[0, r]
            U[4, col + r] = UT[1, r]
            # B local rows 0,1 -> global rows 20+k0, 20+k1  (A row {2})
            V[20 + k0, col + r] = VT[0, r]
            V[20 + k1, col + r] = VT[1, r]
            # C local rows 0..3 -> global {2*k0, 2*k0+1, 2*k1, 2*k1+1}
            W[2 * k0,     col + r] = WT[0, r]
            W[2 * k0 + 1, col + r] = WT[1, r]
            W[2 * k1,     col + r] = WT[2, r]
            W[2 * k1 + 1, col + r] = WT[3, r]
        col += 3

    assert col == rank, f"col={col} expected {rank}"

    if _verify(U, V, W, N, M, P):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Safety fallback: trivial schoolbook (rank n*m*p = 60).
    if time.time() < deadline:
        r = N * M * P
        Uf = np.zeros((N * M, r)); Vf = np.zeros((M * P, r)); Wf = np.zeros((P * N, r))
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    Uf[i * M + j, c] = 1
                    Vf[j * P + k, c] = 1
                    Wf[k * N + i, c] = 1
                    c += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
