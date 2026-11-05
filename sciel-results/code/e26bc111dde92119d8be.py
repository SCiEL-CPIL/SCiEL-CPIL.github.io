"""Matrix multiplication 2x4x4: rank-28 decomposition via 4 independent Strassen 2x2 block products."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    Local indices are 0..3 for 2x2 matrices (row-major)."""
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns

def _build_strassen_28():
    """Build the rank-28 decomposition via 4 independent Strassen 2x2 block products."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_patterns = _strassen_2x2()

    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2),
    ]

    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = a_r + local_r
                global_j = a_c + local_c
                U[global_i * m + global_j][col] += coeff
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_r + local_r
                global_k = b_c + local_c
                V[global_j * p + global_k][col] += coeff
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = c_r + local_r
                global_k = c_c + local_c
                W[global_k * n + global_i][col] += coeff
            col += 1

    return U, V, W

def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c]=1 with trace convention c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, rank):
    """Max abs error between reconstructed tensor and T."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return float(np.max(np.abs(S - T)))

def _try_lower_rank(rank, T, nm, mp, pn, rng, time_limit=10.0):
    """Try to find a valid decomposition at the given rank via random init + gradient descent."""
    t0 = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(20):
        if time.time() - t0 > time_limit:
            break
        U = rng.standard_normal((nm, rank))
        V = rng.standard_normal((mp, rank))
        W = rng.standard_normal((pn, rank))
        
        lr = 0.01
        for step in range(500):
            if time.time() - t0 > time_limit:
                break
            # Compute residual
            S = np.zeros((nm, mp, pn), dtype=float)
            for r in range(rank):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            R = S - T
            
            err = float(np.max(np.abs(R)))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                return U.copy(), V.copy(), W.copy(), err
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                dU[:, r] = np.einsum('abc,bc->a', R, V[:, r] * W[:, r])
                dV[:, r] = np.einsum('abc,ac->b', R, U[:, r] * W[:, r])
                dW[:, r] = np.einsum('abc,ab->c', R, U[:, r] * V[:, r])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < 1e-9:
            return best_U, best_V, best_W, best_err
    
    return best_U, best_V, best_W, best_err

def construct():
    t0 = time.time()
    deadline = 150.0 - 10.0  # safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    T = _build_tensor(n, m, p)

    # Start with rank-28 Strassen decomposition
    U28, V28, W28 = _build_strassen_28()
    U28 = np.array(U28, dtype=float)
    V28 = np.array(V28, dtype=float)
    W28 = np.array(W28, dtype=float)
    
    err28 = _verify(U28, V28, W28, T, 28)
    if err28 > 1e-9:
        # Fallback: schoolbook rank-32
        U32 = np.zeros((nm, 32))
        V32 = np.zeros((mp, 32))
        W32 = np.zeros((pn, 32))
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    U32[a, col] = 1.0
                    V32[b, col] = 1.0
                    W32[c, col] = 1.0
                    col += 1
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}

    # Try to find lower-rank decompositions
    best_result = (U28, V28, W28, 28)
    
    rng = np.random.default_rng(42)
    
    for target_rank in [24, 20, 16, 12]:
        if time.time() > deadline - 30.0:
            break
        U, V, W, err = _try_lower_rank(target_rank, T, nm, mp, pn, rng, time_limit=20.0)
        if U is not None and err < 1e-9:
            best_result = (U, V, W, target_rank)
            break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
