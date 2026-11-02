import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2
        
    def is_valid_decomposition(U, V, W, T, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def try_snap_and_validate(U, V, W, T):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=0.1) and 
                np.allclose(V, V_snap, atol=0.1) and 
                np.allclose(W, W_snap, atol=0.1)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def schoolbook():
        rank = n * m * p
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

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 45 to 49
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        R = target_rank
        found = False
        
        num_restarts = 10 if R == 45 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            random.seed(42 + restart * 1000 + R)
            
            U = np.random.randn(U_dim, R) * 0.5
            V = np.random.randn(V_dim, R) * 0.5
            W = np.random.randn(W_dim, R) * 0.5
            
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 200
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                if it % 20 == 0:
                    valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        found = True
                        break
                
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U -= vU
                V -= vV
                W -= vW
                
                loss = compute_loss(U, V, W)
                
                if loss < prev_loss - 1e-8:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 20:
                        break
                
                if loss < 1e-6:
                    break
            
            if found:
                break
            
            valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
            if valid:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
