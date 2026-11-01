import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = 2, 2, 11

    # ---- Precompute the 44 non-zero (a,b,c) entries of the multiplication tensor ----
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    nz_set = set(tensor_entries)

    def verify(U, V, W, rank):
        """Exact check: returns True if the decomposition is exact."""
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        for a, b, c in tensor_entries:
            if abs(T[a, b, c] - 1.0) > 1e-6:
                return False
        # Check all zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in nz_set:
                        if abs(T[a, b, c]) > 1e-6:
                            return False
        return True

    def fast_error(U, V, W, rank):
        """Cheap error: only checks the 44 non-zero entries (ones)."""
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = 0.0
        for a, b, c in tensor_entries:
            v = T[a, b, c] - 1.0
            err += v * v
        return err

    def try_search(target_rank, time_budget):
        """Random + local search for a rank-target_rank decomposition."""
        if target_rank < 1:
            return None

        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)

        # Sparse random init
        for r in range(target_rank):
            for i in range(n * m):
                if random.random() < 0.3:
                    U[i, r] = random.choice([-1, 1])
            for i in range(m * p):
                if random.random() < 0.12:
                    V[i, r] = random.choice([-1, 1])
            for i in range(p * n):
                if random.random() < 0.12:
                    W[i, r] = random.choice([-1, 1])

        best_error = fast_error(U, V, W, target_rank)
        if best_error < 1e-6 and verify(U, V, W, target_rank):
            return (U.copy(), V.copy(), W.copy())

        max_iters = 20000
        for it in range(max_iters):
            if time.time() > start_time + time_budget:
                break
            if best_error < 1e-8:
                break

            var_type = random.randint(0, 2)
            if var_type == 0:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                old = U[row, col]
                new = random.choice([-1, 0, 1])
                if new == old:
                    continue
                U[row, col] = new
                ne = fast_error(U, V, W, target_rank)
                if ne < best_error:
                    best_error = ne
                else:
                    U[row, col] = old
            elif var_type == 1:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                old = V[row, col]
                new = random.choice([-1, 0, 1])
                if new == old:
                    continue
                V[row, col] = new
                ne = fast_error(U, V, W, target_rank)
                if ne < best_error:
                    best_error = ne
                else:
                    V[row, col] = old
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                old = W[row, col]
                new = random.choice([-1, 0, 1])
                if new == old:
                    continue
                W[row, col] = new
                ne = fast_error(U, V, W, target_rank)
                if ne < best_error:
                    best_error = ne
                else:
                    W[row, col] = old

        if best_error < 1e-6 and verify(U, V, W, target_rank):
            return (U.copy(), V.copy(), W.copy())
        return None

    # ---- Search for lower ranks, spending a bounded time budget ----
    # Try ranks 38 down to 35, with increasing time budgets.
    search_ranks = [38, 37, 36, 35]
    time_budgets = [25, 25, 25, 25]

    for rank, budget in zip(search_ranks, time_budgets):
        if time.time() > start_time + 100:  # leave 40s for fallback
            break
        result = try_search(rank, budget)
        if result is not None:
            U, V, W = result
            # Clean up to integers
            U = np.where(np.abs(U) < 1e-9, 0, np.round(U)).astype(int).tolist()
            V = np.where(np.abs(V) < 1e-9, 0, np.round(V)).astype(int).tolist()
            W = np.where(np.abs(W) < 1e-9, 0, np.round(W)).astype(int).tolist()
            return {"U": U, "V": V, "W": W}

    # ---- Fallback: deterministic rank-39 construction ----
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1

        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1

        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1

        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1

        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1

        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1

        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

    # Last column k=10: schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1

    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1

    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1

    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1

    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1

    return {"U": U, "V": V, "W": W}
