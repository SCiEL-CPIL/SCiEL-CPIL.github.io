import numpy as np
import time

def _verify(U, V, W, n=2, m=3, p=3):
    num_inputs_a = n * m
    num_inputs_b = m * p
    num_outputs = p * n
    T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                T[ia, ib, out_idx] = 1.0
    computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(computed - T)) < 1e-10

def _rank17_solution():
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen for A[:,:2] * B[:2,:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1

    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
    U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1

    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W

def _try_rank16_search(deadline):
    """Attempt to find a rank-16 decomposition via local search."""
    n, m, p = 2, 3, 3
    num_a, num_b, num_c = n*m, m*p, p*n
    rank = 16

    T = np.zeros((num_a, num_b, num_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Initialize from rank-17, zero out one column, then optimize
    U17, V17, W17 = _rank17_solution()
    best_err = np.inf
    best_U, best_V, best_W = None, None, None

    rng = np.random.default_rng(42)

    for trial in range(50):
        if time.time() > deadline - 5:
            break

        # Start from rank-17 with one column zeroed
        U = U17[:, :rank].copy()
        V = V17[:, :rank].copy()
        W = W17[:, :rank].copy()

        # Random perturbation
        scale = 0.1
        U += rng.normal(0, scale, U.shape)
        V += rng.normal(0, scale, V.shape)
        W += rng.normal(0, scale, W.shape)

        # Gradient-free local optimization using coordinate descent
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            # Compute error
            computed = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(computed - T))

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if err < 1e-10:
                break

            # Simple subgradient step (approximate)
            # For each rank component r, adjust to reduce error
            for r in range(rank):
                # Compute contribution of component r
                contrib = np.einsum('ir,jr,kr->ijk', U[:, r:r+1], V[:, r:r+1], W[:, r:r+1])
                residual = T - contrib
                # Gradient w.r.t. U[:,r]
                grad_U = np.einsum('jrk,ijk->ir', V[:, r:r+1], W[:, r:r+1], residual)
                grad_V = np.einsum('irk,ijk->jr', U[:, r:r+1], W[:, r:r+1], residual)
                grad_W = np.einsum('ijr,ijk->kr', U[:, r:r+1], V[:, r:r+1], residual)
                U[:, r] -= lr * grad_U[:, 0]
                V[:, r] -= lr * grad_V[:, 0]
                W[:, r] -= lr * grad_W[:, 0]

        # Check final
        if best_err < 1e-10:
            break

    if best_err < 1e-10 and best_U is not None:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = 2, 3, 3
    num_a, num_b, num_c = n*m, m*p, p*n

    # Try rank 16 first
    result = _try_rank16_search(deadline)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to rank 17
    U, V, W = _rank17_solution()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: rank 18 naive
    rank = 18
    U = np.zeros((num_a, rank))
    V = np.zeros((num_b, rank))
    W = np.zeros((num_c, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
