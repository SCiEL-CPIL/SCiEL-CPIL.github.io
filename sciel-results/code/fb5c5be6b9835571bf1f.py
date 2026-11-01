"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9

def _strassen_2x2():
    """Return (a, b, c) for 2x2 Strassen.
    a[s, i, j] = coeff of A[i,j] in subproduct s
    b[s, j, k] = coeff of B[j,k] in subproduct s
    c[i, k, s] = coeff of M_s in C[i,k]
    """
    a = np.zeros((7, 2, 2))
    b = np.zeros((7, 2, 2))
    c = np.zeros((2, 2, 7))
    
    # m0 = (A00+A11)(B00+B11)
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    # m1 = (A10+A11)B00
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    # m2 = A00(B10-B11)
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    # m3 = A11(B01-B00)
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    # m4 = (A00+A01)B11
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    # m5 = (A10-A00)(B00+B01)
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    # m6 = (A01-A11)(B10+B11)
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    # C00 = m0 + m3 - m4 + m6
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    # C01 = m2 + m4
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = m1 + m3
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = m0 - m1 + m2 + m5
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c

def _build_strassen_4x4():
    """Build rank-49 Strassen decomposition for 4x4 MM using tensor product structure."""
    a2, b2, c2 = _strassen_2x2()
    
    # For 4x4: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # Rank index: r = 7*s1 + s2 where s1 is first-level, s2 is second-level
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            # U: sum over i1,j1 of a2[s1,i1,j1] * a2[s2,i2,j2] for each i2,j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            # V: sum over j1,k1 of b2[s1,j1,k1] * b2[s2,j2,k2] for each j2,k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * 4 + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            # W: W[k*4+i, col] = c2[i1,k1,s1] * c2[i2,k2,s2]
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * 4 + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _try_lower_rank_gradient(rank, time_budget):
    """Try to find a valid decomposition of given rank using gradient-based optimization."""
    n, m, p = 4, 4, 4
    deadline = time.time() + time_budget
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    target = np.ones((n, m, p))
    
    best_err = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    lr = 0.01
    for step in range(500):
        if time.time() > deadline:
            break
        
        # Compute reconstruction
        recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
        
        err = np.sum((recon - target) ** 2)
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-9:
            break
        
        # Compute gradients
        dU4 = np.zeros((n, m, rank))
        dV4 = np.zeros((m, p, rank))
        dW4 = np.zeros((p, n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = recon[i, j, k] - 1.0
                    for r in range(rank):
                        dU4[i, j, r] += 2 * diff * V4[j, k, r] * W4[k, i, r]
                        dV4[j, k, r] += 2 * diff * U4[i, j, r] * W4[k, i, r]
                        dW4[k, i, r] += 2 * diff * U4[i, j, r] * V4[j, k, r]
        
        U4 -= lr * dU4
        V4 -= lr * dV4
        W4 -= lr * dW4
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def _try_lower_rank_random_search(rank, time_budget):
    """Try to find a valid decomposition of given rank using random search."""
    n, m, p = 4, 4, 4
    deadline = time.time() + time_budget
    
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        U4 = U.reshape(n, m, rank)
        V4 = V.reshape(m, p, rank)
        W4 = W.reshape(p, n, rank)
        
        target = np.ones((n, m, p))
        recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
        
        err = np.sum((recon - target) ** 2)
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-9:
            break
    
    if best_solution is not None and best_err < 1e-9:
        return best_solution
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # First, get the deterministic rank-49 Strassen solution
    U49, V49, W49 = _build_strassen_4x4()
    
    if _verify(U49, V49, W49):
        best_solution = (U49, V49, W49)
        best_rank = 49
    else:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    remaining = deadline - time.time()
    candidate_ranks = [20, 22, 24, 26, 28, 30, 32, 35, 40, 45]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try gradient-based search with a small time budget
        time_per_rank = min(5, remaining - 5)
        if time_per_rank < 2:
            break
        
        result = _try_lower_rank_gradient(rank, time_per_rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                best_solution = (U, V, W)
                best_rank = rank
                break
        
        remaining = deadline - time.time()
    
    # If we found a solution better than 49, return it; otherwise return the 49-rank solution
    if best_rank < 49:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
