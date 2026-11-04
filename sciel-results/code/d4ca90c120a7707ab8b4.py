import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    # Precompute index maps: rec index (i,j,k) -> U row, V row, W row
    rec_to_uv = np.zeros((n, m, p), dtype=np.int64)
    rec_to_vw = np.zeros((n, m, p), dtype=np.int64)
    rec_to_wu = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                rec_to_uv[i, j, k] = i * m + j
                rec_to_vw[i, j, k] = j * p + k
                rec_to_wu[i, j, k] = k * n + i

    def compute_rec(U, V, W, rank):
        rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return rec

    def is_valid(U, V, W, rank, tol=1e-7):
        rec = compute_rec(U, V, W, rank)
        return np.max(np.abs(rec - 1.0)) <= tol

    def compute_grads(U, V, W, rank):
        rec = compute_rec(U, V, W, rank)
        err = rec - 1.0
        gU = np.zeros((n * m, rank))
        gV = np.zeros((m * p, rank))
        gW = np.zeros((p * n, rank))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            gU[:, r] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w, optimize=True)
            gV[:, r] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w, optimize=True)
            gW[:, r] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v, optimize=True)
        return gU, gV, gW

    def snap_half(U, V, W):
        return np.round(U * 2.0) / 2.0, np.round(V * 2.0) / 2.0, np.round(W * 2.0) / 2.0

    def try_update(rank, U, V, W, lr, max_iter):
        nonlocal best_solution, best_rank
        U = U.copy(); V = V.copy(); W = W.copy()
        last_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                break
            gU, gV, gW = compute_grads(U, V, W, rank)
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            err = np.max(np.abs(compute_rec(U, V, W, rank) - 1.0))
            if err < 1e-7:
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                break
            if err < last_err:
                last_err = err
            else:
                lr *= 0.5
        # snap to half-integers and verify
        Us, Vs, Ws = snap_half(U, V, W)
        if is_valid(Us, Vs, Ws, rank):
            best_solution = {"U": Us, "V": Vs, "W": Ws}
            best_rank = rank
            return True
        return False

    def try_rank(rank, num_restarts, lr, max_iter):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
        found = False
        for _ in range(num_restarts):
            if time.time() > deadline:
                return found
            # Start from schoolbook, drop (45-rank) random columns, add small noise
            U0, V0, W0 = schoolbook_sol["U"], schoolbook_sol["V"], schoolbook_sol["W"]
            keep = np.random.choice(n * m * p, size=rank, replace=False)
            U = U0[:, keep].copy()
            V = V0[:, keep].copy()
            W = W0[:, keep].copy()
            U += np.random.randn(n * m, rank) * 0.1
            V += np.random.randn(m * p, rank) * 0.1
            W += np.random.randn(p * n, rank) * 0.1
            if try_update(rank, U, V, W, lr, max_iter):
                found = True
                return True
        return found

    # Search from best_rank - 1 downwards
    current_rank = best_rank - 1
    while current_rank >= 20 and time.time() < deadline:
        try_rank(current_rank, num_restarts=3, lr=0.02, max_iter=60)
        current_rank -= 1

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
