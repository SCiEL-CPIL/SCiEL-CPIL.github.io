import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    schoolbook_rank = n * m * p  # 48
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution as fallback
    best_rank = schoolbook_rank
    best_U = np.zeros((nm, schoolbook_rank))
    best_V = np.zeros((mp, schoolbook_rank))
    best_W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c"""
        rank = U.shape[1]
        # For efficiency, check all entries
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum or loop
        max_err = 0.0
        for a in range(nm):
            for b in range(mp):
                # For fixed a,b: need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all c
                # = W @ (U[a,:] * V[b,:]) should equal T[a,b,:]
                vec = U[a, :] * V[b, :]  # (rank,)
                result = W.T @ vec  # (pn,)
                err = np.max(np.abs(result - T[a, b, :]))
                if err > max_err:
                    max_err = err
        return max_err < tol, max_err
    
    # Try to find lower rank solutions using structured approaches
    # Key insight: exploit the Kronecker product structure
    # n=2, m=4=2*2, p=6=2*3
    # We can think of this as combining smaller multiplications
    
    # Approach: Use random search with gradient-based optimization
    # Minimize ||sum_r u_r ⊗ v_r ⊗ w_r - T||_F^2
    
    np.random.seed(42)
    random.seed(42)
    
    def random_search(rank, max_iters=200, lr=0.01):
        """Random initialization + gradient descent to minimize reconstruction error"""
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Flatten T for computation
        T_flat = T.flatten()  # (nm*mp*pn,)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current reconstruction
            # R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Gradient w.r.t. U[a,r] = sum_{b,c} (R[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
            
            # Compute error tensor
            R = np.einsum('ar,br,cr->abc', U, V, W)
            err = R - T  # (nm, mp, pn)
            
            # Gradients
            # dL/dU[a,r] = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', err, V, W)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error
            R = np.einsum('ar,br,cr->abc', U, V, W)
            err_val = np.linalg.norm(R - T)
            if err_val < best_err:
                best_err = err_val
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate
            if err_val > best_err * 1.5:
                lr *= 0.5
            elif err_val < best_err * 0.8:
                lr = min(lr * 1.2, 0.1)
        
        return best_U, best_V, best_W, best_err
    
    # Try various ranks
    for rank in range(7, min(best_rank, 35)):
        if time.time() > deadline:
            break
        
        attempts = 3 if rank > 15 else 5
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            U_sol, V_sol, W_sol, err = random_search(rank)
            
            if err < 1e-6:
                # Found a valid solution!
                best_rank = rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break
        
        if best_rank < schoolbook_rank:
            break  # Found improvement, stop searching higher ranks
    
    # If we found a good solution, try to polish it
    if best_rank < schoolbook_rank:
        # High-precision polish
        U_pol = best_U.copy()
        V_pol = best_V.copy()
        W_pol = best_W.copy()
        
        for it in range(50):
            if time.time() > deadline:
                break
            R = np.einsum('ar,br,cr->abc', U_pol, V_pol, W_pol)
            err = R - T
            grad_U = np.einsum('abc,br,cr->ar', err, V_pol, W_pol)
            grad_V = np.einsum('abc,ar,cr->br', err, U_pol, W_pol)
            grad_W = np.einsum('abc,ar,br->cr', err, U_pol, V_pol)
            
            U_pol -= 0.01 * grad_U
            V_pol -= 0.01 * grad_V
            W_pol -= 0.01 * grad_W
        
        err_final = np.linalg.norm(np.einsum('ar,br,cr->abc', U_pol, V_pol, W_pol) - T)
        if err_final < 1e-6:
            best_U, best_V, best_W = U_pol, V_pol, W_pol
    
    # Convert to required format (list of lists)
    rank = best_U.shape[1]
    U_out = [[float(best_U[a, r]) for r in range(rank)] for a in range(nm)]
    V_out = [[float(best_V[b, r]) for r in range(rank)] for b in range(mp)]
    W_out = [[float(best_W[c, r]) for r in range(rank)] for c in range(pn)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
