"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
_START = time.time()
_LIMIT = 135.0


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    return bool(np.allclose(D, T, atol=1e-9))


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_rank36():
    """Attempt to find a rank-36 decomposition via optimization."""
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Initialize with schoolbook (first 40 columns) and set rest to small random.
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # Columns 40+ are zero, but we only have 36 columns, so this won't work.
    # Let me initialize differently.
    
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Add schoolbook structure to help convergence.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, (i * m + j) % rank] += 1
                V[j * p + k, (j * p + k) % rank] += 1
                W[k * n + i, (k * n + i) % rank] += 1
    
    lr = 0.001
    for step in range(20000):
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = D - T
        loss = np.sum(diff ** 2)
        if loss < 1e-10:
            break
        if step % 5000 == 0 and time.time() > _START + _LIMIT - 30:
            break
        
        dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Round to half-integers.
    for factor in [2, 1]:
        U_r = np.round(U * factor) / factor
        V_r = np.round(V * factor) / factor
        W_r = np.round(W * factor) / factor
        D = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
        if np.allclose(D, T, atol=1e-9):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None


def construct():
    deadline = _START + _LIMIT
    best = _schoolbook()
    best_rank = 40

    if time.time() < deadline - 40:
        r36 = _try_rank36()
        if r36 is not None and _verify(r36["U"], r36["V"], r36["W"]):
            best = r36
            best_rank = 36

    if not _verify(best["U"], best["V"], best["W"]):
        best = _schoolbook()

    return best
# EVOLVE-BLOCK-END
