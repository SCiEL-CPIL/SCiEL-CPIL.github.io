"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)  
    # W: (p*n, rank) = (6, rank)
    
    def tensor_residual(U, V, W):
        """Compute residual efficiently using matrix operations."""
        rank = U.shape[1]
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Compute the reconstructed tensor: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Using einsum for efficiency
        # 'ijr,jkr,kir->ijk'
        reconstructed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # The target tensor is all ones
        target = np.ones((n, m, p))
        
        # Return sum of squared errors
        return np.sum((reconstructed - target) ** 2)

    def is_half_integer(arr, tol=1e-6):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact and has half-integer entries."""
        if not (is_half_integer(U, 1e-5) and is_half_integer(V, 1e-5) and is_half_integer(W, 1e-5)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def als_decompose(target_rank, max_iters=100, tol=1e-12, seed=0):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Target tensor is all ones
        T = np.ones((n, m, p))
        
        prev_error = np.inf
        for iter_ in range(max_iters):
            if time.time() > deadline - 5:
                return None, None, None, None
                
            # Update U: For each (i,j), solve for U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # We want to minimize sum_k (T[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
                    # This is linear in U[i,j,:].
                    # Let x = U[i,j,:].
                    # Minimize ||T[i,j,:] - A x||^2 where A[r, k] = V[j,k,r] * W[k,i,r]
                    A = np.zeros((target_rank, p))
                    for k in range(p):
                        A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update V: For each (j,k), solve for V[j,k,:]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((target_rank, n))
                    for i in range(n):
                        A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update W: For each (k,i), solve for W[k,i,:]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((target_rank, m))
                    for j in range(m):
                        A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
            
            error = tensor_residual(U, V, W)
            
            if error < tol:
                break
                
            # Early stopping if error is not decreasing
            if error > prev_error * 1.01 and iter_ > 10:
                break
            prev_error = error
        
        return U, V, W, error

    def try_rank(target_rank, max_attempts=10):
        """Try to find a rank-minimal decomposition for given target_rank."""
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            seed = 42 + target_rank * 1000 + attempt
            U, V, W, error = als_decompose(target_rank, max_iters=50, tol=1e-10, seed=seed)
            
            if U is None:
                return False
            
            if error < 1e-6:
                # Try to snap to half-integers
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                
                if verify_solution(U_r, V_r, W_r, 1e-6):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = target_rank
                    return True
                else:
                    # If rounding fails, try the raw solution (though it must be half-integer)
                    # The problem requires half-integer entries.
                    # If the raw solution is very close to half-integers but not exactly, 
                    # verify_solution will fail. 
                    # We can try a tighter polish.
                    pass
            
        return False

    # Try ranks from low to high
    # Literature suggests R(2,3,3) is 14 or 15. Schoolbook is 18.
    for target_rank in range(14, rank_sb):
        if time.time() > deadline - 20:
            break
        found = try_rank(target_rank, max_attempts=20)
        if found:
            # Found a solution, try to go lower if time permits
            if time.time() > deadline - 30:
                break
            # Continue loop to try lower ranks? No, we are going up.
            # If we found 14, we can't go lower than 14 easily. 
            # But we should check if 13 is possible? Unlikely.
            # Let's break if we find a low rank.
            break

    return best
# EVOLVE-BLOCK-END
