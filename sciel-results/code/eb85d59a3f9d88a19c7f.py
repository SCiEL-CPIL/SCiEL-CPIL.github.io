import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def _loss_and_grad(U, V, W, n, m, p, rank):
    """
    Compute loss and gradient for the tensor decomposition.
    Loss = 0.5 * sum_{i,k,a,b,b',c} (T_comp - T_target)^2
    T_comp[i,k,a,b,b',c] = sum_r U[a*m+b, r] * V[b'*p+c, r] * W[k*n+i, r]
    T_target[i,k,a,b,b',c] = delta(a,i) * delta(b,b') * delta(c,k)
    """
    # Reshape to 4D tensors for efficient computation?
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # We need to compute the contraction.
    
    # Let's compute the tensor T_comp of shape (n, p, n, m, m, p)
    # T_comp[i, k, a, b, bp, c] = sum_r U[a*m+b, r] * V[bp*p+c, r] * W[k*n+i, r]
    
    # This is equivalent to:
    # Let U4 = U.T reshaped to (R, n, m) -> U4[r, a, b]
    # Let V4 = V.T reshaped to (R, m, p) -> V4[r, bp, c]
    # Let W4 = W.T reshaped to (R, p, n) -> W4[r, k, i]
    
    # T_comp[i, k, a, b, bp, c] = sum_r U4[r, a, b] * V4[r, bp, c] * W4[r, k, i]
    
    # We can compute this by iterating r? No, that's slow for gradient.
    # We can use einsum.
    # T_comp = np.einsum('rab,rbc,rki->ikabpc', U4, V4, W4)
    # But n=4, m=4, p=5, R~80.
    # U4: (80, 4, 4)
    # V4: (80, 4, 5)
    # W4: (80, 5, 4)
    # Output: (4, 5, 4, 4, 4, 5) -> 4*5*4*4*4*5 = 6400 elements.
    # This is small enough to compute explicitly.
    
    U4 = U.T.reshape(rank, n, m)
    V4 = V.T.reshape(rank, m, p)
    W4 = W.T.reshape(rank, p, n)
    
    # Compute T_comp
    # T_comp[i, k, a, b, bp, c]
    # Using einsum: 'rab,rbc,rki->ikabpc'
    # Note: np.einsum might be slow if called frequently in gradient? 
    # But for R=80, it's 80 * (4*4) * (4*5) * (5*4) operations? No, it's a sum over r.
    # Complexity: n*p*n*m*m*p * R = 6400 * 80 = 512k ops. Very fast.
    
    T_comp = np.einsum('rab,rbc,rki->ikabpc', U4, V4, W4, optimize=True)
    
    # T_target
    T_target = np.zeros((n, p, n, m, m, p))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                T_target[i, k, i, b, b, k] = 1.0
    
    residual = T_comp - T_target
    loss = 0.5 * np.sum(residual ** 2)
    
    # Gradients
    # dL/dU4[r,a,b] = sum_{i,k,bp,c} residual[i,k,a,b,bp,c] * V4[r,bp,c] * W4[r,k,i]
    # dL/dV4[r,bp,c] = sum_{i,k,a,b} residual[i,k,a,b,bp,c] * U4[r,a,b] * W4[r,k,i]
    # dL/dW4[r,k,i] = sum_{a,b,bp,c} residual[i,k,a,b,bp,c] * U4[r,a,b] * V4[r,bp,c]
    
    # Using einsum for gradients
    # dU4 = einsum('ikabpc,rbc,rki->rab', residual, V4, W4)
    # dV4 = einsum('ikabpc,rab,rki->rbc', residual, U4, W4)
    # dW4 = einsum('ikabpc,rab,rbc->rki', residual, U4, V4)
    
    dU4 = np.einsum('ikabpc,rbc,rki->rab', residual, V4, W4, optimize=True)
    dV4 = np.einsum('ikabpc,rab,rki->rbc', residual, U4, W4, optimize=True)
    dW4 = np.einsum('ikabpc,rab,rbc->rki', residual, U4, V4, optimize=True)
    
    # Flatten gradients to match U, V, W shapes
    dU = dU4.reshape(rank, n*m).T
    dV = dV4.reshape(rank, m*p).T
    dW = dW4.reshape(rank, p*n).T
    
    return loss, (dU, dV, dW)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Schoolbook initialization
    U = np.zeros((n*m, base_rank))
    V = np.zeros((m*p, base_rank))
    W = np.zeros((p*n, base_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = _loss_and_grad(best_U, best_V, best_W, n, m, p, base_rank)[0]
    
    # Try to reduce rank
    # Start from base_rank - 1 down to some lower bound
    # We will use L-BFGS-B to minimize the loss for a fixed rank R.
    # If we can drive loss to ~0, we have a valid decomposition of rank R.
    
    # Strategy: 
    # 1. Take the schoolbook solution.
    # 2. Try to project it to lower rank? Or just optimize random inits for lower rank?
    # Optimization from scratch for R < base_rank is hard (non-convex).
    # Better: Start with a valid high-rank solution, and try to "prune" or optimize towards lower rank?
    # Actually, L-BFGS-B works best if we start close to a solution.
    # If we start with random U,V,W of rank R, the loss is likely huge and we might get stuck.
    
    # Alternative: Use the schoolbook solution as a "basis".
    # For a target rank R, we can try to find a linear combination of the schoolbook columns that works?
    # No, the columns are not linearly dependent in a way that allows simple reduction for general matrices.
    
    # Let's try optimizing random initializations for specific target ranks.
    # We will prioritize ranks that are theoretically possible or close to known bounds.
    # For 4x4x5, schoolbook is 80.
    # Known best for 4x4x4 is 49. For 4x4x5, likely around 50-60?
    
    target_ranks = [79, 75, 70, 65, 60, 55, 50, 49, 48, 47, 46, 45]
    
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try a few random restarts for this rank
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
                
            # Random init
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
            
            # Combine into a single vector for L-BFGS-B
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def fun(x):
                U_x = x[:n*m*R].reshape(n*m, R)
                V_x = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_x = x[n*m*R + m*p*R :].reshape(p*n, R)
                loss, grad = _loss_and_grad(U_x, V_x, W_x, n, m, p, R)
                grad_flat = np.concatenate([grad[0].flatten(), grad[1].flatten(), grad[2].flatten()])
                return loss, grad_flat
            
            try:
                from scipy.optimize import minimize
                res = minimize(fun, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    # Check if it's a valid decomposition
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Verify with random matrices
                    if _verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist()):
                        best_U = U_sol
                        best_V = V_sol
                        best_W = W_sol
                        best_loss = res.fun
                        found = True
                        break # Move to next lower rank
            except Exception:
                pass
                
        if found:
            continue # Try lower rank
        else:
            # If we failed at this rank, lower ranks might be harder?
            # Or maybe we just need more iterations.
            # Let's stop trying lower ranks if we fail at a higher one?
            # Not necessarily, but usually rank reduction is monotonic in difficulty.
            # However, sometimes a specific rank is a "barrier".
            # Let's just break if we fail at a rank significantly lower than schoolbook?
            # No, let's just continue to next rank in list, but reduce attempts if time is short.
            pass

    # If we didn't find anything better, return schoolbook
    # Ensure we return the best found
    if best_loss < 1e-6:
         return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: Return the schoolbook solution (which is always valid)
    # The schoolbook solution has rank 80.
    # If optimization failed, we still have the initial schoolbook U,V,W.
    # But wait, if we tried R=79 and failed, best_U is still schoolbook (rank 80).
    # We should return the best valid solution found.
    
    # The schoolbook solution is valid.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=5):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        
        # Efficient verification using einsum
        # C_comp[i,k] = sum_r (sum_ab U[a*m+b,r] A[a,b]) * (sum_bc V[b*p+c,r] B[b,c]) * W[k*n+i,r]
        
        # Let's do it explicitly for correctness in verification
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    ap = 0.0
                    for a in range(n):
                        for b in range(m):
                            ap += U[a*m+b, r] * A[a, b]
                    bp = 0.0
                    for b in range(m):
                        for c in range(p):
                            bp += V[b*p+c, r] * B[b, c]
                    val += ap * bp * W[k*n+i, r]
                C_comp[i, k] = val
        
        if not np.allclose(C_comp, C_exp, atol=1e-4):
            return False
    return True
