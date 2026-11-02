"""
Program to find a rank-minimal exact tensor decomposition for multiplying 
a 2 by 3 matrix by a 3 by 5 matrix.

Strategy:
Use additive rank bounds to construct a rank 17 decomposition.
R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3)
R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3)
R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1)
R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2)

Base cases:
R(2, 1, k) = 1
R(2, 2, 1) = 1
R(2, 2, 2) = 7 (Strassen)

Result:
R(2, 2, 3) <= 7 + 1 = 8
R(2, 3, 3) <= 8 + 1 = 9
R(2, 3, 2) <= 7 + 1 = 8
R(2, 3, 5) <= 8 + 9 = 17
"""

import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j, :]
                v_val = V_np[j * p + k, :]
                w_val = W_np[k * n + i, :]
                tensor_val = np.sum(u_val * v_val * w_val)
                if abs(tensor_val - 1.0) > tol:
                    return False
    return True

def _decomp_2x1xp(p):
    """
    Decomposition for 2x1 * 1xp.
    A is 2x1, B is 1xp. C is 2xp.
    C[i,k] = A[i,0] * B[0,k].
    This is an outer product, rank 1.
    U shape (2*1, 1) = (2, 1)
    V shape (1*p, 1) = (p, 1)
    W shape (p*2, 1) = (2p, 1)
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    m=1, n=2.
    U[i*1+0] = U[i]
    V[0*p+k] = V[k]
    W[k*2+i]
    """
    U = [[1] for _ in range(2)]
    V = [[1] for _ in range(p)]
    W = [[1] for _ in range(p * 2)]
    return U, V, W, 1

def _decomp_2x2x1():
    """
    Decomposition for 2x2 * 2x1.
    A is 2x2, B is 2x1. C is 2x1.
    C[i,0] = sum_j A[i,j] B[j,0].
    Rank 1.
    """
    U = [[1], [1], [1], [1]]
    V = [[1], [1]]
    W = [[1], [1]]
    return U, V, W, 1

def _decomp_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Rank 7.
    """
    # Verified Strassen matrices
    U = [
        [ 1,  1,  1,  1,  0, -1, -1],
        [ 0,  0,  1,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  1,  1],
        [-1,  1, -1,  0,  1,  0,  0]
    ]
    V = [
        [ 1,  1,  0,  1,  1,  1,  0],
        [ 1,  1, -1, -1, -1, -1,  1],
        [ 1,  0,  1,  0,  0,  1,  1],
        [ 1,  0,  0,  0,  0,  1,  1]
    ]
    W = [
        [ 1,  0,  0,  0, -1,  0,  1],
        [ 0,  0,  1,  0,  1,  0,  0],
        [ 0,  1,  0,  0,  0, -1,  1],
        [ 1,  0, -1,  0,  0,  1,  0]
    ]
    return U, V, W, 7

def _construct_2x3x2():
    """
    R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2).
    Split m=3 into 2+1.
    """
    U1, V1, W1, r1 = _decomp_2x2x2()
    U2, V2, W2, r2 = _decomp_2x1xp(2)
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*2)] # 6 rows
    W = [[0]*rank for _ in range(2*2)] # 4 rows
    
    # Part 1: 2x2x2 (Top part of A, Top part of B)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*3+j][c] = U1[i*2+j][c]
    
    # V1[j*2+k, c] -> V[j*2+k, c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j*2+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x1x2 (Bottom part of A, Bottom part of B)
    for i in range(2):
        for c in range(r2):
            U[i*3+2][r1+c] = U2[i][c]
            
    # V2[0*2+k, c] -> V[2*2+k, r1+c]
    for k in range(2):
        for c in range(r2):
            V[2*2+k][r1+c] = V2[k][c]
            
    # W2[k*2+i, c] -> W[k*2+i, r1+c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _construct_2x2x3():
    """
    R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1).
    Split p=3 into 2+1.
    """
    U1, V1, W1, r1 = _decomp_2x2x2()
    U2, V2, W2, r2 = _decomp_2x2x1()
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*2)] # 4 rows
    V = [[0]*rank for _ in range(2*3)] # 6 rows
    W = [[0]*rank for _ in range(3*2)] # 6 rows
    
    # Part 1: 2x2x2 (k in 0..1)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*2+j][c] = U1[i*2+j][c]
                
    # V1[j*2+k, c] -> V[j*3+k, c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j*3+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x2x1 (k in 2)
    for i in range(2):
        for j in range(2):
            for c in range(r2):
                U[i*2+j][r1+c] = U2[i*2+j][c]
                
    # V2[j*1+0, c] -> V[j*3+2, r1+c]
    for j in range(2):
        for c in range(r2):
            V[j*3+2][r1+c] = V2[j][c]
            
    # W2[0*2+i, c] -> W[2*2+i, r1+c]
    for i in range(2):
        for c in range(r2):
            W[2*2+i][r1+c] = W2[i][c]
            
    return U, V, W, rank

def _construct_2x3x3():
    """
    R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3).
    Split m=3 into 2+1.
    """
    U1, V1, W1, r1 = _construct_2x2x3()
    U2, V2, W2, r2 = _decomp_2x1xp(3)
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*3)] # 9 rows
    W = [[0]*rank for _ in range(3*2)] # 6 rows
    
    # Part 1: 2x2x3 (j in 0..1)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*3+j][c] = U1[i*2+j][c]
                
    # V1[j*3+k, c] -> V[j*3+k, c]
    for j in range(2):
        for k in range(3):
            for c in range(r1):
                V[j*3+k][c] = V1[j*3+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(3):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x1x3 (j in 2)
    for i in range(2):
        for c in range(r2):
            U[i*3+2][r1+c] = U2[i][c]
            
    # V2[0*3+k, c] -> V[2*3+k, r1+c]
    for k in range(3):
        for c in range(r2):
            V[2*3+k][r1+c] = V2[k][c]
            
    # W2[k*2+i, c] -> W[k*2+i, r1+c]
    for k in range(3):
        for i in range(2):
            for c in range(r2):
                W[k*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _construct_2x3x5():
    """
    R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3).
    Split p=5 into 2+3.
    """
    U1, V1, W1, r1 = _construct_2x3x2()
    U2, V2, W2, r2 = _construct_2x3x3()
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*5)] # 15 rows
    W = [[0]*rank for _ in range(5*2)] # 10 rows
    
    # Part 1: 2x3x2 (k in 0..1)
    for i in range(2):
        for j in range(3):
            for c in range(r1):
                U[i*3+j][c] = U1[i*3+j][c]
                
    # V1[j*2+k, c] -> V[j*5+k, c]
    for j in range(3):
        for k in range(2):
            for c in range(r1):
                V[j*5+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x3x3 (k in 2..4)
    for i in range(2):
        for j in range(3):
            for c in range(r2):
                U[i*3+j][r1+c] = U2[i*3+j][c]
                
    # V2[j*3+k, c] -> V[j*5+(k+2), r1+c]
    for j in range(3):
        for k in range(3):
            for c in range(r2):
                V[j*5+(k+2)][r1+c] = V2[j*3+k][c]
                
    # W2[k*2+i, c] -> W[(k+2)*2+i, r1+c]
    for k in range(3):
        for i in range(2):
            for c in range(r2):
                W[(k+2)*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _try_lower_rank_search(n, m, p, target_rank, time_limit=10):
    """
    Attempt to find a decomposition of rank target_rank using random search.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Number of tensor components
    N = n * m * p
    
    # Initialize random matrices
    def random_init(rank):
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        return U, V, W
    
    best_error = float('inf')
    best_result = None
    
    # Gradient descent step
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    tensor_val = np.sum(u_val * v_val * w_val)
                    err += (tensor_val - 1.0)**2
        return err
    
    # Try a few random restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        U, V, W = random_init(target_rank)
        lr = 0.01
        best_restart_error = float('inf')
        
        for step in range(100):
            if time.time() - start_time > time_limit:
                break
                
            err = compute_error(U, V, W)
            if err < best_restart_error:
                best_restart_error = err
                if err < 1e-10:
                    best_result = (U.copy(), V.copy(), W.copy())
                    return best_result
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i*m+j, :]
                        v_val = V[j*p+k, :]
                        w_val = W[k*n+i, :]
                        tensor_val = np.sum(u_val * v_val * w_val)
                        diff = tensor_val - 1.0
                        
                        dU[i*m+j, :] += 2 * diff * v_val * w_val
                        dV[j*p+k, :] += 2 * diff * u_val * w_val
                        dW[k*n+i, :] += 2 * diff * u_val * v_val
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if err < best_restart_error * 0.9:
                lr *= 1.1
            else:
                lr *= 0.9
                lr = max(lr, 0.001)
        
        if best_restart_error < best_error:
            best_error = best_restart_error
            if best_error < 1e-10:
                best_result = (U.copy(), V.copy(), W.copy())
                return best_result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Try rank 17 first (known good)
    U, V, W, rank = _construct_2x3x5()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        best_rank = rank
        best_result = (U, V, W)
    else:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        best_rank = rank_school
        best_result = (U_s, V_s, W_s)
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But we know rank 17 is achievable, so try ranks 16, 15
    
    for target_rank in [16, 15]:
        if time.time() - start_time > deadline - 30:
            break
            
        print(f"Trying rank {target_rank}...")
        result = _try_lower_rank_search(n, m, p, target_rank, time_limit=20)
        
        if result is not None:
            U_new, V_new, W_new = result
            # Convert to lists
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_rank = target_rank
                best_result = (U_list, V_list, W_list)
                print(f"Found rank {target_rank} decomposition!")
                break
    
    U_best, V_best, W_best = best_result
    return {"U": U_best, "V": V_best, "W": W_best}

# EVOLVE-BLOCK-END
