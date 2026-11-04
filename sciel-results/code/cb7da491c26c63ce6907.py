import numpy as np
import time
import random

def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 145.0
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                triples.append((a, b, c))
    
    # Schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r, (a, b, c) in enumerate(triples):
            U[a, r] = 1.0
            V[b, r] = 1.0
            W[c, r] = 1.0
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = get_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def compute_error(U, V, W):
        # Approximate T and compute L2 error
        # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.linalg.norm(T_approx - T)
    
    def verify_and_update(U, V, W, rank, tol=1e-9):
        nonlocal best_U, best_V, best_W, best_rank
        if rank >= best_rank:
            return False
        # Check if entries are close to integers or half-integers
        # and if the decomposition is exact
        T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(T_approx - T))
        if err > tol:
            return False
        
        # Round to nearest half-integer or integer
        # The problem statement allows integer/half-integer entries
        # We check if rounding preserves the validity
        scales = [2, 1]
        for s in scales:
            U_r = np.round(U * s) / s
            V_r = np.round(V * s) / s
            W_r = np.round(W * s) / s
            T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
            if np.max(np.abs(T_r - T)) < 1e-8:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                return True
        return False
    
    # Try to find rank < 60
    # Lower bound is np (15). Realistically for 3x4x5, it's hard.
    # We will search for rank 30 down to 20 as a compromise, 
    # or just try a few random restarts at rank 59, 58...
    # To be safe and distinct, we use a simple local search / random perturbation from schoolbook
    
    current_rank = best_rank
    # We try to reduce rank by 1 at a time using a perturbation approach
    # This is structurally different from the seed's L-BFGS on random initializations.
    
    for target_rank in range(best_rank - 1, max(15, best_rank - 10), -1):
        if time.time() > deadline:
            break
            
        # Try to fix rank target_rank by perturbing the schoolbook solution
        # and projecting/optimizing
        # Since schoolbook is "full", we can try to zero out one column and optimize the rest?
        # No, that changes the structure.
        
        # Let's do a few random restarts at this target_rank with gradient descent
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook structure but random signs/mixes
            # Strategy: Start with schoolbook, then randomly mix two terms into one?
            # Or just random sparse init.
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = T_approx - T
                
                # Gradients
                # dL/dU[a,r] = sum_b,c diff[a,b,c] * V[b,r] * W[c,r]
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if np.linalg.norm(diff) < 1e-9:
                    break
            
            if verify_and_update(U, V, W, target_rank):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
