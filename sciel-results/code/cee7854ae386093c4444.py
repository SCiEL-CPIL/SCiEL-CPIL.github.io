"""Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Schoolbook fallback: rank 80
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build target tensor for verification
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        est = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(est - target)) < 1e-9
    
    def try_rank_r(r, rng, max_iters=500):
        """Try to find a rank-r decomposition using iterative refinement."""
        if r >= best_rank:
            return None
        
        # Initialize with random factors
        U = rng.standard_normal((n*m, r))
        V = rng.standard_normal((m*p, r))
        W = rng.standard_normal((p*n, r))
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current estimate
            est = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(est - target))
            
            if err < 1e-9:
                return U, V, W
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # If error is small, try Newton-style correction
            if err < 0.1:
                # Compute gradient and apply correction
                # d/dU: sum over b,c of (est-target)[a,b,c] * V[b,:] * W[c,:]
                resid = est - target
                dU = np.einsum('abc,br,cr->ar', resid, V, W)
                dV = np.einsum('abc,ar,cr->br', resid, U, W)
                dW = np.einsum('abc,ar,br->cr', resid, U, V)
                
                # Step size
                step = 0.01
                U -= step * dU
                V -= step * dV
                W -= step * dW
            else:
                # Random perturbation to escape local minima
                U += 0.01 * rng.standard_normal((n*m, r))
                V += 0.01 * rng.standard_normal((m*p, r))
                W += 0.01 * rng.standard_normal((p*n, r))
        
        return (best_U, best_V, best_W) if best_err < 1e-9 else None
    
    # Try randomized iterative approach for ranks from 79 down to 65
    for r in range(79, 64, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(42 + r * 100 + trial)
            result = try_rank_r(r, rng, max_iters=300)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: try a more structured approach using the block decomposition
    # 4x4x4 (rank 49 known) + 4x4x1 (rank 16) = 65
    # But we need explicit factors. Let's try a simpler reduction.
    
    # Try rank 79 by merging schoolbook terms that share structure
    # Terms (i,j,k) and (i',j',k') can potentially merge if there's a linear relation
    
    # For now, return the best result found
    return best_result
# EVOLVE-BLOCK-END
