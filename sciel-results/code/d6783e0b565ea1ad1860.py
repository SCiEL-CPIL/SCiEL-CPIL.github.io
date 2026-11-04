import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s (and 150s) limit

    n, m, p = 3, 4, 6

    def build_target():
        target = np.zeros((n * m, m * p, p * n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
        return target

    TARGET = build_target()

    def verify_arrays(Ua, Va, Wa, atol=1e-9):
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, TARGET, atol=atol)

    def get_rank66():
        """Deterministic rank-66 decomposition: 6x(1x2*2x2) + 6x(Strassen 2x2*2x2)."""
        rank = 66
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            U[A0, col] = 1; V[B00, col] = 1; W[C0, col] = 1
            col += 1
            U[A1, col] = 1; V[B10, col] = 1; W[C0, col] = 1
            col += 1
            U[A0, col] = 1; V[B01, col] = 1; W[C1, col] = 1
            col += 1
            U[A1, col] = 1; V[B11, col] = 1; W[C1, col] = 1
            col += 1

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            U[A00, col] = 1; U[A11, col] = 1
            V[B00, col] = 1; V[B11, col] = 1
            W[C00, col] = 1; W[C11, col] = 1
            col += 1
            U[A10, col] = 1; U[A11, col] = 1
            V[B00, col] = 1
            W[C10, col] = 1; W[C11, col] = -1
            col += 1
            U[A00, col] = 1
            V[B01, col] = 1; V[B11, col] = -1
            W[C01, col] = 1; W[C11, col] = 1
            col += 1
            U[A11, col] = 1
            V[B10, col] = 1; V[B00, col] = -1
            W[C00, col] = 1; W[C10, col] = 1
            col += 1
            U[A00, col] = 1; U[A01, col] = 1
            V[B11, col] = 1
            W[C00, col] = -1; W[C01, col] = 1
            col += 1
            U[A10, col] = 1; U[A00, col] = -1
            V[B00, col] = 1; V[B01, col] = 1
            W[C11, col] = 1
            col += 1
            U[A01, col] = 1; U[A11, col] = -1
            V[B10, col] = 1; V[B11, col] = 1
            W[C00, col] = 1
            col += 1

        for j in range(3):
            bc = 2 * j
            add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
            add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
            add_strassen(1, 0, 0, bc, 1, 2 * j)
            add_strassen(1, 2, 2, bc, 1, 2 * j)

        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W

    # ---- Rank-1 peeling: deterministic structured sweeps ----
    # A column is peelable if removing it keeps the tensor exact. We sweep in
    # order of "exposure": rank-4 (1x2*2x2) columns are loosely coupled to the
    # Strassen core, so we peel those first; interior Strassen columns last.

    def peel_singles(Ua, Va, Wa, order, max_iter=400):
        Ua = Ua.copy(); Va = Va.copy(); Wa = Wa.copy()
        cur = Ua.shape[1]
        present = list(range(cur))
        for _ in range(max_iter):
            if time.time() > deadline - 4.0:
                break
            if not present:
                break
            removed = False
            for col in order:
                if time.time() > deadline - 4.0:
                    break
                if col not in present:
                    continue
                mask = [i for i in present if i != col]
                if verify_arrays(Ua[:, mask], Va[:, mask], Wa[:, mask]):
                    Ua, Va, Wa = Ua[:, mask], Va[:, mask], Wa[:, mask]
                    present.remove(col)
                    removed = True
                    break
            if not removed:
                break
        return Ua, Va, Wa, len(present)

    def peel_pairs(Ua, Va, Wa, max_iter=120):
        Ua = Ua.copy(); Va = Va.copy(); Wa = Wa.copy()
        cur = Ua.shape[1]
        for _ in range(max_iter):
            if time.time() > deadline - 4.0:
                break
            if cur <= 1:
                break
            removed = False
            for c1 in range(cur - 1, 0, -1):
                if time.time() > deadline - 4.0:
                    break
                for c2 in range(c1 - 1, -1, -1):
                    if time.time() > deadline - 4.0:
                        break
                    mask = [i for i in range(cur) if i != c1 and i != c2]
                    if verify_arrays(Ua[:, mask], Va[:, mask], Wa[:, mask]):
                        Ua, Va, Wa = Ua[:, mask], Va[:, mask], Wa[:, mask]
                        cur -= 2
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
        return Ua, Va, Wa, cur

    def peel_triples(Ua, Va, Wa, max_iter=60):
        Ua = Ua.copy(); Va = Va.copy(); Wa = Wa.copy()
        cur = Ua.shape[1]
        for _ in range(max_iter):
            if time.time() > deadline - 4.0:
                break
            if cur <= 2:
                break
            removed = False
            for c1 in range(cur - 1, 1, -1):
                if time.time() > deadline - 4.0:
                    break
                for c2 in range(c1 - 1, 0, -1):
                    if time.time() > deadline - 4.0:
                        break
                    for c3 in range(c2 - 1, -1, -1):
                        if time.time() > deadline - 4.0:
                            break
                        mask = [i for i in range(cur) if i != c1 and i != c2 and i != c3]
                        if verify_arrays(Ua[:, mask], Va[:, mask], Wa[:, mask]):
                            Ua, Va, Wa = Ua[:, mask], Va[:, mask], Wa[:, mask]
                            cur -= 3
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
            if not removed:
                break
        return Ua, Va, Wa, cur

    U66, V66, W66 = get_rank66()
    assert verify_arrays(U66, V66, W66), "Base rank-66 invalid!"

    best_rank = 66
    best_U, best_V, best_W = U66.copy(), V66.copy(), W66.copy()

    # Peel order: rank-4 (1x2*2x2) columns are 0..23 (6 blocks * 4), then Strassen 24..65.
    # Peel the loosely-coupled 1x2*2x2 columns first (highest exposure), then Strassen.
    order_loose = list(range(24)) + list(range(24, 66))

    # Phase 1: deterministic single-column peel, loose-first
    if time.time() < deadline - 40.0:
        U1, V1, W1, r1 = peel_singles(U66, V66, W66, order_loose)
        if r1 < best_rank:
            best_rank = r1
            best_U, best_V, best_W = U1.copy(), V1.copy(), W1.copy()

    # Phase 2: reverse-order single peel (Strassen-first) to catch different reductions
    if time.time() < deadline - 30.0:
        U2, V2, W2, r2 = peel_singles(U66, V66, W66, list(reversed(order_loose)))
        if r2 < best_rank:
            best_rank = r2
            best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()

    # Phase 3: pair peeling from the best
    if time.time() < deadline - 20.0:
        U3, V3, W3, r3 = peel_pairs(best_U, best_V, best_W)
        if r3 < best_rank:
            best_rank = r3
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()

    # Phase 4: triple peeling
    if time.time() < deadline - 12.0:
        U4, V4, W4, r4 = peel_triples(best_U, best_V, best_W)
        if r4 < best_rank:
            best_rank = r4
            best_U, best_V, best_W = U4.copy(), V4.copy(), W4.copy()

    # Phase 5: final single peel pass to polish
    if time.time() < deadline - 4.0 and best_rank > 55:
        U5, V5, W5, r5 = peel_singles(best_U, best_V, best_W, list(range(best_rank)))
        if r5 < best_rank:
            best_rank = r5
            best_U, best_V, best_W = U5.copy(), V5.copy(), W5.copy()

    # Final verification; fallback to known-good rank-66
    if not verify_arrays(best_U, best_V, best_W):
        best_U, best_V, best_W = U66.copy(), V66.copy(), W66.copy()
        best_rank = 66

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
