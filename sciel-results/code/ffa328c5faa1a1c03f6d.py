import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try ranks from 44 down to 37 (or lower)
    # It is unlikely that rank 45 is not minimal for 3x3x5, but let's check 44, 43, etc.
    # We will attempt to find a decomposition of rank R.
    # If we fail, we keep the best found (which might be 45).
    
    # We will try a few specific ranks below 45.
    # If the true rank is 45, these will fail, and we return 45.
    # If the true rank is 44, we might find it.
    
    target_ranks = [44, 43, 42, 41, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 3
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            try:
                result = search_rank_gradient(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_gradient(n, m, p, rank, deadline):
    """
    Uses Gradient Descent with Momentum to minimize the Frobenius norm of the tensor decomposition error.
    """
    # Try multiple different initializations
    initializations = [
        ('small_random', 0.1),
        ('medium_random', 0.5),
        ('structured', None),  # Will use structured initialization
    ]
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init_type, scale in initializations:
        if time.time() > deadline - 10:
            break
            
        if init_type == 'structured':
            # Structured initialization: use schoolbook-like patterns with some randomness
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize with some non-zero entries to help convergence
            for r in range(rank):
                # Randomly select some positions
                for _ in range(3):
                    ui = np.random.randint(0, n * m)
                    U[ui, r] = np.random.choice([-1, 1]) * 0.5
                    
                for _ in range(3):
                    vj = np.random.randint(0, m * p)
                    V[vj, r] = np.random.choice([-1, 1]) * 0.5
                    
                for _ in range(3):
                    wk = np.random.randint(0, p * n)
                    W[wk, r] = np.random.choice([-1, 1]) * 0.5
        else:
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale
        
        # Learning rate and momentum
        lr = 0.005
        momentum = 0.9
        
        U_vel = np.zeros_like(U)
        V_vel = np.zeros_like(V)
        W_vel = np.zeros_like(W)
        
        max_iter = 1500
        last_error = float('inf')
        plateau_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Compute T = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.sum((T - 1.0)**2)
            
            # Check for improvement
            if error < last_error - 1e-10:
                last_error = error
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Early stopping if plateau
            if plateau_count > 200:
                break
            
            if error < 1e-10:
                # Try snapping
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                U3s = U_snap.reshape(n, m, rank)
                V3s = V_snap.reshape(m, p, rank)
                W3s = W_snap.reshape(p, n, rank)
                Ts = np.einsum('ijr,jkr,kir->ijk', U3s, V3s, W3s)
                snap_error = np.sum((Ts - 1.0)**2)
                
                if snap_error < 1e-8:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # Compute Gradients
            D = 2.0 * (T - 1.0)
            
            dU = np.einsum('ijk,jkr,kir->ijr', D, V3, W3).reshape(n*m, rank)
            dV = np.einsum('ijk,ijr,kir->jkr', D, U3, W3).reshape(m*p, rank)
            dW = np.einsum('ijk,ijr,jkr->kir', D, U3, V3).reshape(p*n, rank)
            
            # Update with momentum
            U_vel = momentum * U_vel - lr * dU
            V_vel = momentum * V_vel - lr * dV
            W_vel = momentum * W_vel - lr * dW
            
            U += U_vel
            V += V_vel
            W += W_vel
        
        # Track best
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Final check on best found
    if best_U is not None:
        U3 = best_U.reshape(n, m, rank)
        V3 = best_V.reshape(m, p, rank)
        W3 = best_W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-6:
            # Try snapping with different granularities
            for granularity in [2, 4, 8]:
                U_snap = np.round(best_U * granularity) / granularity
                V_snap = np.round(best_V * granularity) / granularity
                W_snap = np.round(best_W * granularity) / granularity
                
                U3s = U_snap.reshape(n, m, rank)
                V3s = V_snap.reshape(m, p, rank)
                W3s = W_snap.reshape(p, n, rank)
                Ts = np.einsum('ijr,jkr,kir->ijk', U3s, V3s, W3s)
                snap_error = np.sum((Ts - 1.0)**2)
                
                if snap_error < 1e-8:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None
