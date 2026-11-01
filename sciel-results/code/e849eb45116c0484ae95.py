"""Matrix multiplication 2,3,10 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the target tensor T[(i,j),(j',k),(k,i)] = delta(j,j')
    # U index: i*m + j, V index: j'*p + k, W index: k*n + i
    # T[u, v, w] = 1 if u's j-index == v's j'-index
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W decompose T exactly."""
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=1e-9)
    
    # Schoolbook decomposition (rank = n*m*p = 60)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank_found = best_rank
    
    # Try Winograd-style structured reduction
    # For 2x3 * 3x10, we can try to exploit the small n=2 dimension
    # by using a Kronecker-like structure
    
    # Approach: Try to find a decomposition with rank < 60 by exploiting
    # the structure. The key is that for matrix multiplication C = A*B where
    # A is n x m and B is m x p, we have C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For n=2, m=3, p=10, let's try a structured approach:
    # Group the computation by the j-index (middle dimension)
    
    # Try a "block" approach: for each j in 0..2, we compute contributions
    # to all C[i,k] from A[i,j] * B[j,k]
    
    # Winograd's approach for small matrices uses polynomial interpolation
    # For 2x3*3x10, we can try to reduce the number of multiplications
    
    # Let's try a more systematic approach: use the fact that we can
    # precompute linear combinations
    
    # For each "rank-1" term r, we have:
    # U[:,r] is a vector in R^(n*m) = R^6
    # V[:,r] is a vector in R^(m*p) = R^30  
    # W[:,r] is a vector in R^(p*n) = R^20
    
    # The constraint is: sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
    
    # This is a 3-way tensor decomposition problem. Let's try to find
    # a lower rank decomposition using structured random search with
    # gradient-based refinement.
    
    # First, let's try a simpler approach: check if rank 59 is possible
    # by trying to find one redundant column in the schoolbook decomposition
    
    # For each column r in the schoolbook, check if we can express it
    # as a linear combination of the others (in the tensor sense)
    
    # This is equivalent to checking if the tensor has rank <= 59
    # which is hard in general, but let's try a heuristic
    
    # Better approach: Use the fact that for matrix multiplication,
    # there are known lower bounds and constructions.
    
    # For 2x3 * 3x10:
    # - Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # - Upper bound (schoolbook): 60
    # - Winograd's bound: O(n^2.376 * m * p) but that's asymptotic
    
    # Let's try a structured random search with refinement
    
    def random_search_with_refinement(target_rank, n_trials=50, max_iter=200):
        """Try to find a decomposition of given rank using random init + gradient refinement."""
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with some structure
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Try to optimize to minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
            # Using a simple gradient descent approach
            
            # For efficiency, we'll use a coordinate descent approach
            # where we update one tensor at a time
            
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update U: for each u, minimize sum_v,w (T[u,v,w] - sum_r U[u,r]*V[v,r]*W[w,r])^2
                # This is a linear system in U[u,:]
                # Let's use a simplified approach: just do a few random updates
                
                # Actually, for speed, let's just check if the random init is close
                # and return early if not promising
                
                if it % 10 == 0:
                    # Quick check
                    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_recon - T)
                    if err < 1e-6:
                        return U.copy(), V.copy(), W.copy()
                
                # Simple update: nudge U, V, W in random directions
                # and keep if error decreases
                U_old = U.copy()
                V_old = V.copy()
                W_old = W.copy()
                
                U += np.random.randn(n*m, target_rank) * lr
                V += np.random.randn(m*p, target_rank) * lr
                W += np.random.randn(p*n, target_rank) * lr
                
                T_recon_new = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
                err_new = np.linalg.norm(T_recon_new - T)
                
                T_recon_old = np.einsum('ur,vr,wr->uvw', U_old, V_old, W_old, optimize=True)
                err_old = np.linalg.norm(T_recon_old - T)
                
                if err_new > err_old:
                    U, V, W = U_old, V_old, W_old
                    lr *= 0.95
            
            # Final check
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
            if np.allclose(T_recon, T, atol=1e-6):
                return U.copy(), V.copy(), W.copy()
        
        return None
    
    # Try to find lower rank decompositions
    # Start from rank 59 and go down
    for target_rank in range(best_rank_found - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() > deadline:
            break
        
        result = random_search_with_refinement(target_rank, n_trials=5, max_iter=50)
        if result is not None:
            U_new, V_new, W_new = result
            if verify(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank_found = target_rank
                break  # Found a lower rank, stop
    
    # If we didn't find anything better, return schoolbook
    if best_rank_found == best_rank:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
