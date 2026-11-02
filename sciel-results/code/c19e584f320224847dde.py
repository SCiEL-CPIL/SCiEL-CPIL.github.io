import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
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
    
    def validate_full(U, V, W, atol=1e-6):
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
        return False, None, None, None
    
    def compute_loss(U, V, W, T_target):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T_target
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
    
    # Build the 1x5x5 sub-tensor
    # For row i of output, the sub-tensor has:
    # U: 5 rows (j=0..4), V: 25 rows (j*5+k), W: 5 rows (k=0..4)
    # T_sub[j, j*5+k, k] = 1
    sub_T = np.zeros((m, m*p, p))
    for j in range(m):
        for k in range(p):
            sub_T[j, j*p+k, k] = 1.0
    
    # Try to find rank-5 decomposition of the 1x5x5 sub-tensor
    # Using structured initialization and optimization
    
    best_sub_solution = None
    best_sub_rank = m * p  # 25
    
    # First, try a direct rank-5 construction
    # The 1x5x5 tensor T_sub[j, j*5+k, k] = 1 can be viewed as:
    # For each (j,k), we need U[j,r]*V[j*5+k,r]*W[k,r] to sum to 1
    
    # Try rank 5 with various structured patterns
    for R in range(5, 16):
        if time.time() > deadline - 30:
            break
        
        found = False
        num_restarts = 10 if R <= 8 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 25:
                break
            
            random.seed(42 + restart * 104729 + R * 31)
            
            U_sub = np.zeros((m, R))
            V_sub = np.zeros((m*p, R))
            W_sub = np.zeros((p, R))
            
            # Initialize: assign each (j,k) pair to a rank position
            all_pairs = [(j, k) for j in range(m) for k in range(p)]
            
            if R == 5:
                # Try structured: each rank position r handles pairs where j+k ≡ r (mod 5)
                for (j, k) in all_pairs:
                    r = (j + k) % R
                    U_sub[j, r] += 1.0
                    V_sub[j*p+k, r] += 1.0
                    W_sub[k, r] += 1.0
            elif R == 10:
                # Each rank position handles a subset
                for (j, k) in all_pairs:
                    r = (j * 2 + k) % R
                    U_sub[j, r] += 1.0
                    V_sub[j*p+k, r] += 1.0
                    W_sub[k, r] += 1.0
            else:
                # Random assignment
                for (j, k) in all_pairs:
                    r = random.randint(0, R-1)
                    U_sub[j, r] += 1.0
                    V_sub[j*p+k, r] += 1.0
                    W_sub[k, r] += 1.0
            
            # Gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U_sub)
            vV = np.zeros_like(V_sub)
            vW = np.zeros_like(W_sub)
            
            max_iter = 500
            patience = 100
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
                
                loss = compute_loss(U_sub, V_sub, W_sub, sub_T)
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                if loss < 1e-10:
                    break
            
            # Try to snap to half-integers and validate
            U_snap = snap_to_half_integers(U_sub)
            V_snap = snap_to_half_integers(V_sub)
            W_snap = snap_to_half_integers(W_sub)
            
            if (np.allclose(U_sub, U_snap, atol=1e-4) and 
                np.allclose(V_sub, V_snap, atol=1e-4) and 
                np.allclose(W_sub, W_snap, atol=1e-4)):
                prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.allclose(prod, sub_T, atol=1e-4):
                    best_sub_solution = (U_snap, V_snap, W_snap)
                    best_sub_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # If we found a sub-solution, construct the full 2x5x5 solution
    if best_sub_solution:
        U_sub, V_sub, W_sub = best_sub_solution
        R = best_sub_rank
        
        # Full U: 10 rows = [U_sub (5 rows for i=0), U_sub (5 rows for i=1)]
        U_full = np.vstack([U_sub, U_sub])
        # Full V: 25 rows = V_sub
        V_full = V_sub.copy()
        # Full W: 10 rows, W index = k*n + i = 2k + i
        # W_full[2k] = W_sub[k] for i=0
        # W_full[2k+1] = W_sub[k] for i=1
        W_full = np.zeros((W_dim, R))
        for k in range(p):
            W_full[2*k, :] = W_sub[k, :]
            W_full[2*k+1, :] = W_sub[k, :]
        
        # Validate full solution
        prod = np.einsum('ar,br,cr->abc', U_full, V_full, W_full, optimize=True)
        if np.allclose(prod, T, atol=1e-4):
            U_snap = snap_to_half_integers(U_full)
            V_snap = snap_to_half_integers(V_full)
            W_snap = snap_to_half_integers(W_full)
            prod2 = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
            if np.allclose(prod2, T, atol=1e-4):
                U_list = [[float(x) for x in row] for row in U_snap]
                V_list = [[float(x) for x in row] for row in V_snap]
                W_list = [[float(x) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try direct optimization on the full tensor with various ranks
    for R in range(10, 26):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 12:
                break
            
            random.seed(999 + restart * 7919 + R * 13)
            
            U_full = np.zeros((U_dim, R))
            V_full = np.zeros((V_dim, R))
            W_full = np.zeros((W_dim, R))
            
            # Initialize with random structured patterns
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = random.randint(0, R-1)
                        U_full[i*m+j, r] += 1.0
                        V_full[j*p+k, r] += 1.0
                        W_full[k*n+i, r] += 1.0
            
            # Gradient descent
            lr = 0.005
            momentum = 0.9
            vU = np.zeros_like(U_full)
            vV = np.zeros_like(V_full)
            vW = np.zeros_like(W_full)
            
            max_iter = 300
            patience = 80
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                prod = np.einsum('ar,br,cr->abc', U_full, V_full, W_full, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V_full, W_full, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U_full, W_full, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U_full, V_full, optimize=True)
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U_full -= vU
                V_full -= vV
                W_full -= vW
                
                loss = compute_loss(U_full, V_full, W_full, T)
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                if loss < 1e-10:
                    break
            
            # Validate
            U_snap = snap_to_half_integers(U_full)
            V_snap = snap_to_half_integers(V_full)
            W_snap = snap_to_half_integers(W_full)
            
            if (np.allclose(U_full, U_snap, atol=1e-4) and 
                np.allclose(V_full, V_snap, atol=1e-4) and 
                np.allclose(W_full, W_snap, atol=1e-4)):
                prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.allclose(prod, T, atol=1e-4):
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
