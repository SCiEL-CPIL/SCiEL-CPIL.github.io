import time
import numpy as np

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 115.0  # safety margin inside the 120 s wall-clock


def _target_tensor(n, m, p):
    """Expected 3-way tensor of matrix multiplication (indices a=i*m+j, b=j*p+k, c=k*n+i)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _build_trivial(n, m, p):
    """Rank n*m*p decomposition: one rank-1 term per (i,j,k)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    return U, V, W


def _reconstruct(U, V, W):
    """Sum over rank-1 terms: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.int64)
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return T


def _is_valid(U, V, W, target):
    return np.array_equal(_reconstruct(U, V, W), target)


def _search_structured(n, m, p, target):
    """
    Structural search for a lower-rank exact decomposition.

    Each rank-1 term is (u_vec, v_vec, w_vec) with entries in {-1,0,1} and a scalar weight.
    We try small 'basis' rank-1 patterns (low Hamming weight) and search over which subset
    of basis patterns (with integer weights) sums exactly to the target tensor.

    This is a different modelling family than pure random search: we restrict the search
    to a small candidate pool of structured patterns and solve a (sparse) integer
    reconstruction problem.
    """
    # Candidate pool of rank-1 patterns. Each pattern is (u,v,w) with entries in {-1,0,1}.
    # We include:
    #  - the 56 trivial unit patterns
    #  - a number of structured low-weight patterns
    rng = np.random.default_rng(12345)

    pool = []
    # unit patterns
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m, dtype=np.int8); u[i * m + j] = 1
                v = np.zeros(m * p, dtype=np.int8); v[j * p + k] = 1
                w = np.zeros(p * n, dtype=np.int8); w[k * n + i] = 1
                pool.append((u, v, w))

    # structured low-weight patterns: pick a small set of support indices for each factor
    n_struct = 400
    for _ in range(n_struct):
        if time.time() > _DEADLINE - 15:
            break
        su = rng.choice(n * m, size=rng.integers(1, 4), replace=False)
        sv = rng.choice(m * p, size=rng.integers(1, 4), replace=False)
        sw = rng.choice(p * n, size=rng.integers(1, 4), replace=False)
        u = np.zeros(n * m, dtype=np.int8); u[su] = rng.integers(-1, 2, size=len(su))
        v = np.zeros(m * p, dtype=np.int8); v[sv] = rng.integers(-1, 2, size=len(sv))
        w = np.zeros(p * n, dtype=np.int8); w[sw] = rng.integers(-1, 2, size=len(sw))
        if np.any(u) and np.any(v) and np.any(w):
            pool.append((u, v, w))

    # Precompute each pattern's tensor (rank-1 contribution, weight 1)
    L = len(pool)
    T_pool = np.zeros((L, n * m, m * p, p * n), dtype=np.int64)
    for r in range(L):
        u, v, w = pool[r]
        T_pool[r] = np.einsum('a,b,c->abc', u, v, w)

    target_flat = target.reshape(-1)
    T_pool_flat = T_pool.reshape(L, -1)

    # Greedy / sparse reconstruction: try to represent target as sum of few pool tensors
    # with integer coefficients in {-3,-2,-1,0,1,2,3}.
    # Use a simple greedy selection: repeatedly pick the pool pattern that best reduces
    # the residual, with coefficient from {-1,0,1} (and small multiples).
    best_U = np.zeros((n * m, 0), dtype=int)
    best_V = np.zeros((m * p, 0), dtype=int)
    best_W = np.zeros((p * n, 0), dtype=int)
    best_rank = n * m * p  # trivial rank

    def _try_greedy(max_terms, coeff_range):
        nonlocal best_U, best_V, best_W, best_rank
        residual = target_flat.copy()
        used = []
        for step in range(max_terms):
            if time.time() > _DEADLINE - 10:
                return
            if np.all(residual == 0):
                break
            # For each unused pool pattern, find best coefficient in coeff_range
            best_r = -1
            best_c = 0
            best_err = np.sum(residual * residual)
            for r in range(L):
                if any(u[0] == r for u in used):
                    continue
                for c in coeff_range:
                    if c == 0:
                        continue
                    cand = residual - c * T_pool_flat[r]
                    err = np.sum(cand * cand)
                    if err < best_err:
                        best_err = err
                        best_r = r
                        best_c = c
            if best_r == -1 or best_c == 0:
                break
            residual = residual - best_c * T_pool_flat[best_r]
            used.append((best_r, best_c))
        if np.all(residual == 0) and len(used) < best_rank:
            # build U,V,W from used patterns with their coefficients
            k = len(used)
            U = np.zeros((n * m, k), dtype=int)
            V = np.zeros((m * p, k), dtype=int)
            W = np.zeros((p * n, k), dtype=int)
            for col, (r, c) in enumerate(used):
                U[:, col] = pool[r][0] * c
                V[:, col] = pool[r][1]
                W[:, col] = pool[r][2]
            if _is_valid(U, V, W, target):
                best_U, best_V, best_W = U, V, W
                best_rank = k

    _try_greedy(40, [-3, -2, -1, 1, 2, 3])
    if time.time() < _DEADLINE - 10:
        _try_greedy(60, [-2, -1, 1, 2])

    if best_rank < n * m * p:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    target = _target_tensor(n, m, p)

    U, V, W = _build_trivial(n, m, p)  # guaranteed valid fallback

    found = _search_structured(n, m, p, target)
    if found is not None:
        U, V, W = found

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    target = _target_tensor(n, m, p)
    if _is_valid(U, V, W, target):
        print("Valid decomposition, rank =", U.shape[1])
    else:
        print("Invalid decomposition")
