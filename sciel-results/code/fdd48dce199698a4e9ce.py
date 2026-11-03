import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
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
    
    def validate(U, V, W, atol=1e-6):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None
    
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
    
    # Key insight: The 2x5x5 tensor decomposes into two independent 1x5x5 sub-tensors
    # Row 0 of output: W indices 0,2,4,6,8 (even)
    # Row 1 of output: W indices 1,3,5,7,9 (odd)
    # So rank(2x5x5) = 2 * rank(1x5x5)
    
    # Build the 1x5x5 sub-tensor for row i
    def build_subtensor():
        # U: 5 rows (j=0..4), V: 25 rows (j*5+k), W: 5 rows (k=0..4)
        sub_T = np.zeros((m, m*p, p))
        for j in range(m):
            for k in range(p):
                sub_T[j, j*p+k, k] = 1.0
        return sub_T
    
    sub_T = build_subtensor()
    
    # Search for minimal rank of 1x5x5 tensor
    # Start from schoolbook rank 25 and try to reduce
    best_sub_solution = None
    best_sub_rank = m * p  # 25
    
    # Try ranks from 25 down to 15
    for target_rank in range(m * p, 14, -1):
        if time.time() > deadline - 30:
            break
            
        R = target_rank
        found = False
        
        # Multiple restarts
        num_restarts = 3 if R >= 20 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 25:
                break
            
            random.seed(12345 + restart * 7919 + R * 31)
            
            # Initialize with random structured patterns
            U_sub = np.zeros((m, R))
            V_sub = np.zeros((m*p, R))
            W_sub = np.zeros((p, R))
            
            # Strategy: each rank position handles a subset of the (j,k) pairs
            all_pairs = [(j, k) for j in range(m) for k in range(p)]
            assignments = [random.choice(range(R)) for _ in range(len(all_pairs))]
            
            for (j, k), rank_pos in zip(all_pairs, assignments):
                U_sub[j, rank_pos] += 1.0
                V_sub[j*p+k, rank_pos] += 1.0
                W_sub[k, rank_pos] += 1.0
            
            # Gradient descent
            lr = 0.005
            momentum = 0.9
            vU = np.zeros_like(U_sub)
            vV = np.zeros_like(V_sub)
            vW = np.zeros_like(W_sub)
            
            max_iter = 400
            patience = 80
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 20:
                    break
                
                prod = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sub, optimize=True)
                diff = prod - sub_T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V_sub, W_sub, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U_sub, W_sub, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U_sub, V_sub, optimize=True)
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U_sub -= vU
                V_sub -= vV
                W_sub -= vW
                
                loss = np.sum(diff * diff)
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                if loss < 1e-8:
                    break
            
            # Validate
            U_snap = snap_to_half_integers(U_sub)
            V_snap = snap_to_half_integers(V_sub)
            W_snap = snap_to_half_integers(W_sub)
            
            if (np.allclose(U_sub, U_snap, atol=1e-6) and 
                np.allclose(V_sub, V_snap, atol=1e-6) and 
                np.allclose(W_sub, W_snap, atol=1e-6)):
                prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.allclose(prod, sub_T, atol=1e-6):
                    best_sub_solution = (U_snap, V_snap, W_snap)
                    best_sub_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # Construct full 2x5x5 solution by stacking the two sub-solutions
    if best_sub_solution:
        U_sub, V_sub, W_sub = best_sub_solution
        R = best_sub_rank
        
        # Full U: 10 rows = [U_sub (5 rows), U_sub (5 rows)]
        U_full = np.vstack([U_sub, U_sub])
        # Full V: 25 rows = V_sub
        V_full = V_sub.copy()
        # Full W: 10 rows = [W_sub (5 rows for i=0), W_sub (5 rows for i=1)]
        # W index for row i is k*2+i, so:
        # W_full[2k] = W_sub[k] for k=0..4
        # W_full[2k+1] = W_sub[k] for k=0..4
        W_full = np.zeros((W_dim, R))
        for k in range(p):
            W_full[2*k, :] = W_sub[k, :]
            W_full[2*k+1, :] = W_sub[k, :]
        
        # Validate full solution
        prod = np.einsum('ar,br,cr->abc', U_full, V_full, W_full, optimize=True)
        if np.allclose(prod, T, atol=1e-6):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback to schoolbook
    return schoolbook()
