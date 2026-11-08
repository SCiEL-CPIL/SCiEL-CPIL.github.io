"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]


def _build_tensors(n, m, p):
    """Build the multiplication tensor and schoolbook factor matrices."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1

    U0 = np.zeros((n * m, n * m * p), dtype=np.int64)
    V0 = np.zeros((m * p, n * m * p), dtype=np.int64)
    W0 = np.zeros((p * n, n * m * p), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, c] = 1
                V0[j * p + k, c] = 1
                W0[k * n + i, c] = 1
                c += 1
    return T, U0, V0, W0


def _verify(U, V, W, T, tol=1e-6):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    n = T.shape[0]
    m = T.shape[1]
    p = T.shape[2]
    rank = U.shape[1]
    acc = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # reshape U column to (n,m), V to (m,p), W to (p,n)
        um = uc.reshape(n, m)
        vp = vc.reshape(m, p)
        wn = wc.reshape(p, n)
        # tensor product: (i,j,k) = um[i,j]*vp[j,k]*wn[k,i]
        for i in range(n):
            for j in range(m):
                if um[i, j] == 0:
                    continue
                for k in range(p):
                    acc[i, j, k] += um[i, j] * vp[j, k] * wn[k, i]
    diff = np.max(np.abs(acc - T.astype(np.float64)))
    return diff < tol, diff


def _random_factor(T, n, m, p, seed):
    """Generate a random rank-1 factor with small integer entries."""
    rng = np.random.RandomState(seed)
    # Support sizes: pick subsets of rows/cols
    # U: (n*m) vector, V: (m*p) vector, W: (p*n) vector
    # We'll create sparse-ish factors
    u = np.zeros(n * m, dtype=np.int64)
    v = np.zeros(m * p, dtype=np.int64)
    w = np.zeros(p * n, dtype=np.int64)

    # Choose support sizes
    su = rng.randint(1, min(4, n * m) + 1)
    sv = rng.randint(1, min(4, m * p) + 1)
    sw = rng.randint(1, min(4, p * n) + 1)

    ui = rng.choice(n * m, su, replace=False)
    vi = rng.choice(m * p, sv, replace=False)
    wi = rng.choice(p * n, sw, replace=False)

    for idx in ui:
        u[idx] = rng.randint(-2, 3)
    for idx in vi:
        v[idx] = rng.randint(-2, 3)
    for idx in wi:
        w[idx] = rng.randint(-2, 3)

    # Ensure at least one nonzero
    if np.all(u == 0):
        u[ui[0]] = 1
    if np.all(v == 0):
        v[vi[0]] = 1
    if np.all(w == 0):
        w[wi[0]] = 1

    return u, v, w


def _factor_contribution(u, v, w, n, m, p):
    """Compute the tensor contribution of one rank-1 factor."""
    um = u.reshape(n, m)
    vp = v.reshape(m, p)
    wn = w.reshape(p, n)
    contrib = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            if um[i, j] == 0:
                continue
            for k in range(p):
                contrib[i, j, k] += um[i, j] * vp[j, k] * wn[k, i]
    return contrib


def _greedy_complete(remaining, T, n, m, p, U_list, V_list, W_list, max_iter=200, seed=0):
    """Try to complete the decomposition greedily from the remaining tensor."""
    rng = np.random.RandomState(seed)
    remaining = remaining.copy()
    U_list = list(U_list)
    V_list = list(V_list)
    W_list = list(W_list)

    for it in range(max_iter):
        if np.all(remaining == 0):
            break
        # Try random factors and pick the one that reduces ||remaining||
        best_factor = None
        best_norm = np.sum(remaining ** 2)
        for trial in range(50):
            u, v, w = _random_factor(T, n, m, p, rng.randint(0, 2**31 - 1))
            contrib = _factor_contribution(u, v, w, n, m, p)
            new_rem = remaining - contrib
            new_norm = np.sum(new_rem ** 2)
            if new_norm < best_norm:
                best_norm = new_norm
                best_factor = (u, v, w)
                best_new_rem = new_rem
        if best_factor is None:
            break
        u, v, w = best_factor
        remaining = best_new_rem
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    if np.all(remaining == 0):
        return True, U_list, V_list, W_list
    return False, U_list, V_list, W_list


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    T, U0, V0, W0 = _build_tensors(n, m, p)
    schoolbook_rank = n * m * p

    # Start with schoolbook as baseline
    best_rank = schoolbook_rank
    best_U = U0.tolist()
    best_V = V0.tolist()
    best_W = W0.tolist()

    # Try to find lower rank via random search + greedy completion
    rng = np.random.RandomState(42)

    # Target: try to get rank as low as possible
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Known results for 3x5x5: try to find something below 75

    # Strategy: randomly generate partial decompositions and try to complete
    max_target_rank = 60  # try to find rank <= 60
    attempts = 0

    for target_rank in range(25, max_target_rank + 1):
        if time.time() > deadline:
            break
        # Try several random initializations for this target rank
        for attempt in range(30):
            if time.time() > deadline:
                break
            attempts += 1
            # Generate target_rank random factors
            U_list = []
            V_list = []
            W_list = []
            remaining = T.copy()

            for c in range(target_rank):
                u, v, w = _random_factor(T, n, m, p, rng.randint(0, 2**31 - 1))
                contrib = _factor_contribution(u, v, w, n, m, p)
                remaining = remaining - contrib
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)

            # Try to complete greedily
            success, U_list, V_list, W_list = _greedy_complete(
                remaining, T, n, m, p, U_list, V_list, W_list,
                max_iter=100, seed=rng.randint(0, 2**31 - 1)
            )

            if success:
                final_rank = len(U_list)
                if final_rank < best_rank:
                    best_rank = final_rank
                    best_U = [u.tolist() for u in U_list]
                    best_V = [v.tolist() for v in V_list]
                    best_W = [w.tolist() for w in W_list]
                    # Verify
                    U_arr = np.array(best_U).T  # (rank, n*m) -> transpose to (n*m, rank)
                    V_arr = np.array(best_V).T
                    W_arr = np.array(best_W).T
                    ok, diff = _verify(U_arr, V_arr, W_arr, T)
                    if ok:
                        break  # Found valid lower rank, try even lower

    # Build final output
    rank = best_rank
    U_out = np.zeros((n * m, rank), dtype=np.int64)
    V_out = np.zeros((m * p, rank), dtype=np.int64)
    W_out = np.zeros((p * n, rank), dtype=np.int64)

    for c in range(rank):
        U_out[:, c] = best_U[c]
        V_out[:, c] = best_V[c]
        W_out[:, c] = best_W[c]

    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
