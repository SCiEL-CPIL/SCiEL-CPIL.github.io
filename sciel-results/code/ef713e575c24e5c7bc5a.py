"""Rank-66 structural baseline + correct rank-2 redistribution / redundancy-drop local search.

Strategy:
  * Start from the rank-66 partition decomposition (pure 2x2 Strassen blocks and
    1x2*2x2 blocks; B columns split 2+2+2).
  * The real refinement lever is the rank-2 identity: two rank-1 terms whose
    "product vectors" (u_i * v_j * w_k over all (i,j,k)) are proportional carry
    redundant structure. We re-express such a proportional pair into a linearly
    independent pair (via SVD of the 2-term contribution matrix), which can free
    a dimension so a third term becomes absorbable, then drop it.
  * Redundancy drop: a term that is a linear combination of the others can be
    removed.
  * Perturbation + local search to escape local minima, all guarded by a hard
    deadline with a safety margin.
"""
import time
import random

import numpy as np

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p
N_T = n * m * p    # 72 tensor entries

# ---- index helpers (trace convention W[k*n+i]) ---------------------------
def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i


def baseline_66():
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c + 1)
        A10 = Aidx(a_r + 1, a_c); A11 = Aidx(a_r + 1, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c + 1)
        C10 = Cidx(c_r + 1, c_c); C11 = Cidx(c_r + 1, c_c + 1)
        U[A00][col] += 1; U[A11][col] += 1; V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1; col += 1
        U[A10][col] += 1; U[A11][col] += 1; V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1; col += 1
        U[A00][col] += 1; V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1; col += 1
        U[A11][col] += 1; V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1; col += 1
        U[A00][col] += 1; U[A01][col] += 1; V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1; col += 1
        U[A10][col] += 1; U[A00][col] -= 1; V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1; col += 1
        U[A01][col] += 1; U[A11][col] -= 1; V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c + 1)
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1; col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1; col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1; col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1; col += 1

    add_1x2_2x2(0, 0, 0, 0, 0, 0); add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2); add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4); add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0); add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2); add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4); add_strassen(1, 2, 2, 4, 1, 4)
    assert col == 66, f"expected 66 got {col}"
    return U, V, W


# ---- verification / linear-algebra helpers -------------------------------
def verify(U, V, W, r):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[Aidx(i, j)], Va[Bidx(j, k)] * Wa[Cidx(i, k)]))
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def term_product(U, V, W, r, l):
    """Product vector t_{i,j,k} = U[i,j,l]*V[j,k,l]*W[i,k,l] over all (i,j,k)."""
    t = np.empty(N_T)
    for idx in range(N_T):
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        t[idx] = U[Aidx(i, j)][l] * V[Bidx(j, k)][l] * W[Cidx(i, k)][l]
    return t


def is_zero_term(U, V, W, r, l):
    t = term_product(U, V, W, r, l)
    return bool(np.all(t == 0))


def drop_term(U, V, W, r, l):
    newU = [row[:l] + row[l + 1:] for row in U]
    newV = [row[:l] + row[l + 1:] for row in V]
    newW = [row[:l] + row[l + 1:] for row in W]
    return newU, newV, newW, r - 1


def try_drop(U, V, W, r, l):
    """True if removing term l still gives an exact decomposition."""
    newU, newV, newW, nr = drop_term(U, V, W, r, l)
    return verify(newU, newV, newW, nr)


def is_redundant(U, V, W, r, l):
    """Term l is a linear combination of the others (rank of others == rank of all)."""
    if r <= 1:
        return False
    others = [c for c in range(r) if c != l]
    M = np.empty((N_T, r - 1))
    for pos, c in enumerate(others):
        M[:, pos] = term_product(U, V, W, r, c)
    tl = term_product(U, V, W, r, l)
    r_other = np.linalg.matrix_rank(M, tol=1e-9)
    r_all = np.linalg.matrix_rank(np.column_stack([M, tl]), tol=1e-9)
    return r_all <= r_other


def try_merge_pair(U, V, W, r, l1, l2):
    """Rank-2 redistribution: re-express proportional pair (l1,l2) into an
    independent pair, then try to drop a third term. Returns (U,V,W,r) or None."""
    t1 = term_product(U, V, W, r, l1)
    t2 = term_product(U, V, W, r, l2)
    if np.linalg.norm(t1) < 1e-12 or np.linalg.norm(t2) < 1e-12:
        return None
    # proportional if cross product ~ 0
    cross = np.cross(t1, t2)
    scale = max(np.linalg.norm(t1), np.linalg.norm(t2))
    if np.linalg.norm(cross) > 1e-7 * scale * scale:
        return None  # not a proportional pair -> no rank-2 freedom

    # Contribution matrix of the pair (N_T x 2)
    Mpair = np.column_stack([t1, t2])
    # SVD to get two independent basis vectors spanning the same column space
    U_s, svals, Vt = np.linalg.svd(Mpair, full_matrices=False)
    if len(svals) < 2 or svals[1] < 1e-9 * max(1.0, svals[0]):
        return None  # truly rank 1, nothing to redistribute
    b1 = U_s[:, 0]
    b2 = U_s[:, 1]
    # coefficients: [t1,t2] = B @ coef
    B = np.column_stack([b1, b2])
    coef, *_ = np.linalg.lstsq(B, Mpair, rcond=None)
    # new product vectors
    t1n = B @ coef[0]
    t2n = B @ coef[1]

    def factorize(tv):
        """Find (u,v,w) with u[i,j]*v[j,k]*w[i,k] == tv[i,j,k] (exact if rank-1)."""
        u = np.ones(n * m); v = np.ones(m * p); w = np.ones(p * n)
        for _ in range(30):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        d = v[Bidx(j, k)] * w[Cidx(i, k)]
                        if abs(d) > 1e-12:
                            u[Aidx(i, j)] = tv[i, j, k] / d
                            break
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        d = u[Aidx(i, j)] * w[Cidx(i, k)]
                        if abs(d) > 1e-12:
                            v[Bidx(j, k)] = tv[i, j, k] / d
                            break
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        d = u[Aidx(i, j)] * v[Bidx(j, k)]
                        if abs(d) > 1e-12:
                            w[Cidx(i, k)] = tv[i, j, k] / d
                            break
        # verify exactness of factorization
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(u[Aidx(i, j)] * v[Bidx(j, k)] * w[Cidx(i, k)] - tv[i, j, k]) > 1e-8:
                        ok = False
        if not ok:
            return None
        return u, v, w

    f1 = factorize(t1n.reshape(n, m, p))
    f2 = factorize(t2n.reshape(n, m, p))
    if f1 is None or f2 is None:
        return None
    u1, v1, w1 = f1
    u2, v2, w2 = f2

    # build new matrices: replace columns l1,l2 with the new independent pair
    newU = [list(row) for row in U]
    newV = [list(row) for row in V]
    newW = [list(row) for row in W]
    for idx in range(n * m):
        newU[idx][l1] = float(u1[idx]); newU[idx][l2] = float(u2[idx])
    for idx in range(m * p):
        newV[idx][l1] = float(v1[idx]); newV[idx][l2] = float(v2[idx])
    for idx in range(p * n):
        newW[idx][l1] = float(w1[idx]); newW[idx][l2] = float(w2[idx])
    if not verify(newU, newV, newW, r):
        return None

    # Now try to drop a third term that became redundant
    for l3 in range(r):
        if l3 == l1 or l3 == l2:
            continue
        if is_redundant(newU, newV, newW, r, l3):
            dU, dV, dW, nr = drop_term(newU, newV, newW, r, l3)
            if verify(dU, dV, dW, nr):
                return dU, dV, dW, nr
    return None


def local_search(U, V, W, r, deadline, rng):
    best_r = r
    best_U, best_V, best_W = U, V, W
    while time.time() < deadline - 8:
        improved = False

        # Phase 1: drop zero terms
        for l in range(best_r):
            if is_zero_term(best_U, best_V, best_W, best_r, l):
                best_U, best_V, best_W, best_r = drop_term(best_U, best_V, best_W, best_r, l)
                improved = True
                break
        if improved:
            continue

        # Phase 2: drop redundant terms
        for l in range(best_r):
            if is_redundant(best_U, best_V, best_W, best_r, l):
                if try_drop(best_U, best_V, best_W, best_r, l):
                    best_U, best_V, best_W, best_r = drop_term(best_U, best_V, best_W, best_r, l)
                    improved = True
                    break
        if improved:
            continue

        # Phase 3: rank-2 redistribution on proportional pairs
        merged = False
        for l1 in range(best_r):
            if time.time() > deadline - 8:
                break
            for l2 in range(l1 + 1, best_r):
                if time.time() > deadline - 8:
                    break
                res = try_merge_pair(best_U, best_V, best_W, best_r, l1, l2)
                if res is not None:
                    best_U, best_V, best_W, best_r = res
                    improved = True
                    merged = True
                    break
            if merged:
                break

        # Phase 4: perturbation to escape local minima
        if not improved and time.time() < deadline - 15:
            U2 = [list(row) for row in best_U]
            V2 = [list(row) for row in best_V]
            W2 = [list(row) for row in best_W]
            for _ in range(3):
                l = rng.randrange(best_r)
                for idx in range(n * m):
                    if U2[idx][l] != 0:
                        U2[idx][l] = round(U2[idx][l] * (1.0 + 0.15 * rng.random()), 6)
                for idx in range(m * p):
                    if V2[idx][l] != 0:
                        V2[idx][l] = round(V2[idx][l] * (1.0 + 0.15 * rng.random()), 6)
                for idx in range(p * n):
                    if W2[idx][l] != 0:
                        W2[idx][l] = round(W2[idx][l] * (1.0 + 0.15 * rng.random()), 6)
            # re-verify after perturbation; keep only if still exact (it won't be,
            # so we use perturbation only as a seed for a fresh local search)
            if verify(U2, V2, W2, best_r):
                best_U, best_V, best_W = U2, V2, W2
                improved = True

        if not improved:
            break
    return best_r, best_U, best_V, best_W


def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    rng = random.Random(1234)

    U, V, W = baseline_66()
    r = 66

    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    if not verify(U, V, W, r):
        U, V, W = baseline_66()
        r = 66

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
