import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        M = np.einsum('ar,cr->acr', U, W)  # (nm, pn, rank)
        T_approx = np.einsum('acr,br->abc', M, V)  # (nm, mp, pn)
        return np.allclose(T_approx, T, atol=tol, rtol=0)
    
    def try_round_and_verify(U, V, W, scale=2):
        """Try rounding to half-integers and verify"""
        for s in [scale, scale*2, scale*4, scale*8]:
            U_r = np.round(U * s) / s
            V_r = np.round(V * s) / s
            W_r = np.round(W * s) / s
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None
    
    # Strategy: Start from schoolbook, try to reduce rank by finding
    # linear dependencies and using gradient-based optimization
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb
    
    # Try ranks from 49 down to 30
    for target_rank in range(rank_sb - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Start from schoolbook with random linear combinations
                # Pick target_rank random linear combinations of the 50 schoolbook terms
                idx = np.random.choice(rank_sb, size=target_rank, replace=False)
                U = U_sb[:, idx].copy()
                V = V_sb[:, idx].copy()
                W = W_sb[:, idx].copy()
            elif restart % 3 == 1:
                # Random initialization with structure
                np.random.seed(100 + restart)
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
            else:
                # Sparse initialization: start with schoolbook, zero out random columns
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                # Add small random perturbation
                U += np.random.randn(nm, target_rank) * 0.01
                V += np.random.randn(mp, target_rank) * 0.01
                W += np.random.randn(pn, target_rank) * 0.01
            
            # Gradient descent with adaptive step size
            lr = 0.05
            best_res = np.inf
            no_improve_count = 0
            
            for step in range(800):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual
                M = np.einsum('ar,cr->acr', U, W)
                T_approx = np.einsum('acr,br->abc', M, V)
                R = T - T_approx
                res = np.sum(R**2)
                
                if res < best_res:
                    best_res = res
                    no_improve_count = 0
                    if best_res < 1e-12:
                        break
                else:
                    no_improve_count += 1
                
                if no_improve_count > 50:
                    lr *= 0.8
                    if lr < 1e-6:
                        break
                
                # Compute gradients
                # dL/dU[a,r] = -2 * sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if res < best_res * 0.99:
                    lr = min(lr * 1.02, 0.5)
                elif res > best_res * 1.1:
                    lr *= 0.7
            
            # Check if we found a good solution
            if best_res < 1e-8:
                # Try to round
                rounded = try_round_and_verify(U, V, W)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if verify(U_r, V_r, W_r):
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = target_rank
                        found = True
                        break
                else:
                    # Use unrounded if it's very close
                    if best_res < 1e-10:
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            # Found a solution at this rank, try to go lower
            continue
        else:
            # If we couldn't find at this rank, stop trying lower
            break
    
    return best_result
