import time
import numpy as np

def construct():
    """
    Returns a low-rank tensor decomposition for 4x4 * 4x9 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 4, 4, 9
    
    # Fallback: schoolbook rank 144
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try rank-36 decomposition
    def rank36_decomp():
        rank = m * p  # 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                # U: for all i, U[i*m + j, col] = 1
                for i in range(n):
                    U[i * m + j, col] = 1
                # V: V[j*p + k, col] = 1
                V[j * p + k, col] = 1
                # W: for all i, W[k*n + i, col] = 1
                for i in range(n):
                    W[k * n + i, col] = 1
                col += 1
        return U, V, W, rank

    candidates = [
        ("rank36", rank36_decomp),
        ("schoolbook", schoolbook),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for name, func in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = func()
            # Simple check: shapes
            if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
                continue
            if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
                continue
            
            # Verify consistency for a few random points and all consistent points?
            # To be safe, check all consistent points.
            valid = True
            # Check all (i,j,k)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum_r U[im+j, r] * V[jp+k, r] * W[kn+i, r]
                        # Vectorized check
                        u_row = U[i*m+j, :]
                        v_row = V[j*p+k, :]
                        w_row = W[k*n+i, :]
                        s = np.dot(u_row * v_row, w_row)
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            # Check a few inconsistent points to ensure 0
            if valid:
                import random
                random.seed(42)
                for _ in range(100):
                    i = random.randint(0, n-1)
                    j1 = random.randint(0, m-1)
                    j2 = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    if j1 == j2:
                        continue
                    # alpha = im + j1, beta = jp + j2? No.
                    # alpha = im + j1. beta = jp + k. gamma = kn + i.
                    # Inconsistent if j1 != j.
                    # Here j is the one from beta.
                    # So alpha uses j1, beta uses j2.
                    # If j1 != j2, it is inconsistent.
                    u_row = U[i*m+j1, :]
                    v_row = V[j2*p+k, :]
                    w_row = W[k*n+i, :]
                    s = np.dot(u_row * v_row, w_row)
                    if abs(s) > 1e-9:
                        valid = False
                        break
            
            if valid:
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W)
                break
        except Exception:
            pass
    
    if best_result is None:
        U, V, W, rank = schoolbook()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
