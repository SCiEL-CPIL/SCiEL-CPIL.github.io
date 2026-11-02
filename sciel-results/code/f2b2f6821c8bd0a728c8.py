"""
Rank-minimal tensor decomposition for 2x2x9 matrix multiplication.
Strategy: Find a rank-9 decomposition for 2x2x3, then use 3 blocks for 2x2x9 (Rank 27).
Fallback to rank-32 (4x Strassen + 1 Schoolbook) if rank-9 search fails.
"""
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank32():
    """
    Build the known rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 schoolbook (col 8).
    Returns (U, V, W) as lists of lists of floats.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b_{0,k0}+b_{1,k1})
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2: (a10+a11) * b_{0,k0}
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3: a00 * (b_{0,k1} - b_{1,k1})
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4: a11 * (b_{1,k0} - b_{0,k0})
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5: (a00+a01) * b_{1,k1}
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6: (a10-a00) * (b_{0,k0}+b_{0,k1})
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7: (a01-a11) * (b_{1,k0}+b_{1,k1})
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    return U, V, W

def _find_2x2x3_rank9(deadline):
    """
    Attempt to find a rank-9 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) for n=2, m=2, p=3 if found, else None.
    """
    n, m, p = 2, 2, 3
    rank = 9
    
    # Target tensor T[i,j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Initialize U, V, W
    # Use a heuristic initialization based on Strassen-like patterns or random
    # Random initialization with multiple restarts
    best_loss = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Scale to ensure products are not too large
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Gradient Descent
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Compute current tensor
            T_calc = np.zeros((n, m, p))
            for r in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            err = T_calc - T
            loss = np.sum(err**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dL/dU_{i,r} = 2 * sum_{j,k} err_{i,j,k} * V_{j,r} * W_{k,r}
            # dL/dV_{j,r} = 2 * sum_{i,k} err_{i,j,k} * U_{i,r} * W_{k,r}
            # dL/dW_{k,r} = 2 * sum_{i,j} err_{i,j,k} * U_{i,r} * V_{j,r}
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(rank):
                # dU
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dU[i, r] += 2 * err[i, j, k] * V[j, r] * W[k, r]
                            
                # dV
                for j in range(m):
                    for i in range(n):
                        for k in range(p):
                            dV[j, r] += 2 * err[i, j, k] * U[i, r] * W[k, r]
                            
                # dW
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            dW[k, r] += 2 * err[i, j, k] * U[i, r] * V[j, r]
                            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic restart of lr or normalization?
            if step % 100 == 0:
                lr *= 0.9

    if best_loss < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find rank-27 solution (3 blocks of rank-9)
    if time.time() < deadline - 10:
        result_2x2x3 = _find_2x2x3_rank9(deadline - 10)
        if result_2x2x3 is not None:
            U3, V3, W3 = result_2x2x3
            # Construct 2x2x9 by repeating the block 3 times
            n, m, p = 2, 2, 9
            rank = 27
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            for block_idx in range(3):
                col_offset = block_idx * 9
                for i in range(n * m):
                    for r in range(9):
                        U[i][col_offset + r] = U3[i][r]
                        
                for j in range(m):
                    for k in range(3):
                        v_idx = j * 3 + k
                        for r in range(9):
                            V[v_idx][col_offset + r] = V3[v_idx][r]
                            
                for k in range(3):
                    for i in range(n):
                        w_idx = k * n + i
                        for r in range(9):
                            W[w_idx][col_offset + r] = W3[w_idx][r]
                            
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            else:
                # If verification fails, it's a false positive, fallback to 32
                pass

    # Fallback to rank-32
    U, V, W = _build_rank32()
    if not _verify(U, V, W, 2, 2, 9):
        # Fallback to full schoolbook rank 36
        rank_fb = 36
        U_fb = [[0.0] * rank_fb for _ in range(4)]
        V_fb = [[0.0] * rank_fb for _ in range(18)]
        W_fb = [[0.0] * rank_fb for _ in range(18)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][c] = 1.0
                    V_fb[j*9+k][c] = 1.0
                    W_fb[k*2+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U, "V": V, "W": W}
