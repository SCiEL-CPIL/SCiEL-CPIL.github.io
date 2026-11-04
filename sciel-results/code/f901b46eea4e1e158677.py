import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
DEADLINE = 140.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition. U:(nm,rank) V:(mp,rank) W:(pn,rank).
    W index convention: W[k*n + i, c]."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    rec = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.allclose(rec, np.ones((n, m, p)), atol=tol)


def _verify_fast(U, V, W, rank, tol=1e-9):
    """Verify assuming arrays already reshaped: U:(n,m,rank) V:(m,p,rank) W:(p,n,rank)."""
    rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return np.allclose(rec, np.ones((N, M, P)), atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2():
    """Rank-7 tensors for 2x2 x 2x2 -> 2x2.
    U[i,j,c], V[j,k,c], W[k,i,c] with i,j,k in {0,1}.
    Returns U,V,W each shape (4,7) flattened row-major (i,j),(j,k),(k,i)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    def setU(i, j, c, v): U[i * 2 + j, c] += v
    def setV(j, k, c, v): V[j * 2 + k, c] += v
    def setW(k, i, c, v): W[k * 2 + i, c] += v

    # m0 = (a00+a11)(b00+b11)
    setU(0, 0, 0, 1); setU(1, 1, 0, 1)
    setV(0, 0, 0, 1); setV(1, 1, 0, 1)
    setW(0, 0, 0, 1); setW(1, 1, 0, 1)
    # m1 = (a10+a11)b00
    setU(1, 0, 1, 1); setU(1, 1, 1, 1)
    setV(0, 0, 1, 1)
    setW(0, 1, 1, 1); setW(1, 1, 1, -1)
    # m2 = a00(b01-b11)
    setU(0, 0, 2, 1)
    setV(0, 1, 2, 1); setV(1, 1, 2, -1)
    setW(1, 0, 2, 1); setW(1, 1, 2, 1)
    # m3 = a11(b10-b00)
    setU(1, 1, 3, 1)
    setV(1, 0, 3, 1); setV(0, 0, 3, -1)
    setW(0, 1, 3, 1); setW(1, 1, 3, 1)
    # m4 = (a00+a01)b11
    setU(0, 0, 4, 1); setU(0, 1, 4, 1)
    setV(1, 1, 4, 1)
    setW(1, 0, 4, -1); setW(1, 1, 4, 1)
    # m5 = (a10-a00)(b00+b01)
    setU(1, 0, 5, 1); setU(0, 0, 5, -1)
    setV(0, 0, 5, 1); setV(0, 1, 5, 1)
    setW(0, 1, 5, 1)
    # m6 = (a01-a11)(b10+b11)
    setU(0, 1, 6, 1); setU(1, 1, 6, -1)
    setV(1, 0, 6, 1); setV(1, 1, 6, 1)
    setW(1, 0, 6, 1)

    rec = np.einsum('ijc,jkc,kic->ijk', U.reshape(2, 2, 7),
                    V.reshape(2, 2, 7), W.reshape(2, 2, 7))
    assert np.allclose(rec, np.ones((2, 2, 2)), atol=1e-12), "Strassen 2x2 broken"
    return U, V, W


def _build_4x4x4_strassen():
    """Rank-28 decomposition for 4x4 x 4x4 -> 4x4 via block Strassen.
    4x4 split into four 2x2 blocks. Block-level Strassen (7 terms),
    each term multiplies two 2x2 blocks (schoolbook rank 4). Total 7*4=28.
    Returns U:(16,28) V:(16,28) W:(16,28)."""
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))

    Uforms = [
        {('A', 0, 0): 1, ('A', 1, 1): 1},
        {('A', 1, 0): 1, ('A', 1, 1): 1},
        {('A', 0, 0): 1},
        {('A', 1, 1): 1},
        {('A', 0, 0): 1, ('A', 0, 1): 1},
        {('A', 1, 0): 1, ('A', 0, 0): -1},
        {('A', 0, 1): 1, ('A', 1, 1): -1},
    ]
    Vforms = [
        {('B', 0, 0): 1, ('B', 1, 1): 1},
        {('B', 0, 0): 1},
        {('B', 0, 1): 1, ('B', 1, 1): -1},
        {('B', 1, 0): 1, ('B', 0, 0): -1},
        {('B', 1, 1): 1},
        {('B', 0, 0): 1, ('B', 0, 1): 1},
        {('B', 1, 0): 1, ('B', 1, 1): 1},
    ]
    Wforms = [
        {('C', 0, 0): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 1, 1): -1},
        {('C', 0, 1): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 0, 0): 1},
        {('C', 0, 1): -1, ('C', 1, 1): 1},
        {('C', 1, 1): 1},
        {('C', 0, 0): 1},
    ]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc
                for (tag, br, bc), coeff in Uforms[s].items():
                    for i2 in range(2):
                        for j2 in range(2):
                            i = br * 2 + i2
                            j = bc * 2 + j2
                            U[i * 4 + j, c] += coeff
                for (tag, br, bc), coeff in Vforms[s].items():
                    for j2 in range(2):
                        for k2 in range(2):
                            j = br * 2 + j2
                            k = bc * 2 + k2
                            V[j * 4 + k, c] += coeff
                for (tag, br, bc), coeff in Wforms[s].items():
                    for k2 in range(2):
                        for i2 in range(2):
                            k = br * 2 + k2
                            i = bc * 2 + i2
                            W[k * 4 + i, c] += coeff

    assert _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 4), "4x4x4 Strassen broken"
    return U, V, W


def _build_4x4x5():
    """Combine 4x4x4 Strassen (rank 28) + boundary 4x4 x 4x1 (rank 16) = rank 44.
    U:(16,44) V:(20,44) W:(20,44)."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_total = 44
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))

    U[:, :rank_core] = U44
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :rank_core] = V44[j * 4 + k, :rank_core]
    W[:16, :rank_core] = W44

    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0  # k=4, i

    assert _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P), "4x4x5 broken"
    return U, V, W


def _direct_column_removal(U, V, W, deadline=None, tol=1e-9, max_rounds=50):
    """Try removing each column and verify directly. Repeat until no more removals.
    This is the key improvement: don't rely on linear dependence heuristics,
    just try and check if the tensor decomposition still holds."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    for round_num in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        removed = False
        # Try removing each column from last to first
        for c in range(rank - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            
            Ut = [row[:c] + row[c + 1:] for row in U]
            Vt = [row[:c] + row[c + 1:] for row in V]
            Wt = [row[:c] + row[c + 1:] for row in W]
            
            if _verify(Ut, Vt, Wt, N, M, P, tol):
                U, V, W = Ut, Vt, Wt
                rank -= 1
                removed = True
                break  # Restart scan after removal
        
        if not removed:
            break
    
    return U, V, W


def _pairwise_column_removal(U, V, W, deadline=None, tol=1e-9):
    """Try removing pairs of columns that might be jointly redundant.
    After single removals stabilize, try pairs."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    for c1 in range(rank - 1, -1, -1):
        if deadline is not None and time.time() > deadline:
            break
        for c2 in range(c1 - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            
            Ut = [row[:c2] + row[c2+1:c1] + row[c1+1:] for row in U]
            Vt = [row[:c2] + row[c2+1:c1] + row[c1+1:] for row in V]
            Wt = [row[:c2] + row[c2+1:c1] + row[c1+1:] for row in W]
            
            if _verify(Ut, Vt, Wt, N, M, P, tol):
                U, V, W = Ut, Vt, Wt
                rank -= 2
                break
    
    return U, V, W


def _triple_column_removal(U, V, W, deadline=None, tol=1e-9):
    """Try removing triples of columns. Only worth it if we have time."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    if rank < 3:
        return U, V, W
    
    for c1 in range(rank - 1, -1, -1):
        if deadline is not None and time.time() > deadline:
            break
        for c2 in range(c1 - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            for c3 in range(c2 - 1, -1, -1):
                if deadline is not None and time.time() > deadline:
                    break
                
                Ut = [row[:c3] + row[c3+1:c2] + row[c2+1:c1] + row[c1+1:] for row in U]
                Vt = [row[:c3] + row[c3+1:c2] + row[c2+1:c1] + row[c1+1:] for row in V]
                Wt = [row[:c3] + row[c3+1:c2] + row[c2+1:c1] + row[c1+1:] for row in W]
                
                if _verify(Ut, Vt, Wt, N, M, P, tol):
                    U, V, W = Ut, Vt, Wt
                    rank -= 3
                    return U, V, W
    
    return U, V, W


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # fallback: schoolbook
    Us, Vs, Ws = _schoolbook(N, M, P)
    assert _verify(Us, Vs, Ws, N, M, P)
    best_rank = N * M * P
    best = {"U": Us, "V": Vs, "W": Ws}

    # structured 4x4x5 (rank 44)
    if time.time() < deadline:
        try:
            U445, V445, W445 = _build_4x4x5()
            if _verify(U445.tolist(), V445.tolist(), N, M, P):
                best_rank = 44
                best = {"U": U445.tolist(), "V": V445.tolist(), "W": W445.tolist()}

                # Direct column removal - the key improvement
                if time.time() < deadline:
                    Ur, Vr, Wr = _direct_column_removal(
                        best["U"], best["V"], best["W"],
                        deadline=deadline - 30.0, max_rounds=50)
                    rr = len(Ur[0])
                    if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                        best_rank = rr
                        best = {"U": Ur, "V": Vr, "W": Wr}

                # Pairwise removal
                if time.time() < deadline:
                    remain = deadline - time.time() - 10.0
                    if remain > 5.0:
                        Ur, Vr, Wr = _pairwise_column_removal(
                            best["U"], best["V"], best["W"],
                            deadline=start + remain)
                        rr = len(Ur[0])
                        if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                            best_rank = rr
                            best = {"U": Ur, "V": Vr, "W": Wr}

                # Triple removal if we have time
                if time.time() < deadline:
                    remain = deadline - time.time() - 5.0
                    if remain > 3.0 and best_rank > 30:
                        Ur, Vr, Wr = _triple_column_removal(
                            best["U"], best["V"], best["W"],
                            deadline=start + remain)
                        rr = len(Ur[0])
                        if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                            best_rank = rr
                            best = {"U": Ur, "V": Vr, "W": Wr}

                # Final round of direct removal after pair/triple
                if time.time() < deadline:
                    remain = deadline - time.time() - 2.0
                    if remain > 1.0:
                        Ur, Vr, Wr = _direct_column_removal(
                            best["U"], best["V"], best["W"],
                            deadline=start + remain, max_rounds=20)
                        rr = len(Ur[0])
                        if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                            best_rank = rr
                            best = {"U": Ur, "V": Vr, "W": Wr}

        except Exception:
            pass

    return best
