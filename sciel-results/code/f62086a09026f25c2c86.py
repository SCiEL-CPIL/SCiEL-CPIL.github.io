import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s deadline, 20s safety margin
    
    # Target tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Strategy: Try ranks from 44 down, use random initialization + local search
    
    # For each candidate rank, try to find U, V, W such that
    # sum_r U[:,r] * V[:,r] * W[:,r] = T
    
    # We'll use a simple approach: random initialization and gradient-free local search
    # This is unlikely to work well for high-dimensional problems, but let's try
    
    # Actually, let's try a smarter approach: exploit the structure
    # The tensor has rank at most 38 (known from literature for similar problems)
    # Let's try to find rank 38, 39, 40, 41, 42, 43, 44
    
    for target_rank in range(44, 37, -1):  # Try 44 down to 38
        if time.time() > deadline - 10:
            break
        
        success = False
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple local search: try to reduce error
            # Error = ||T - sum_r U[:,r] V[:,r] W[:,r]||_F^2
            # This is a non-convex problem, so we'll use a simple approach
            
            # Compute current error
            def compute_error(U, V, W):
                # Efficient computation
                # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # = sum_r (U[a,r] * V[b,r]) * W[c,r]
                # Let's compute this as a tensor
                T_approx = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.sum((T - T_approx)**2)
            
            current_error = compute_error(U, V, W)
            
            # Try a few random perturbations
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Random perturbation
                perturbation = np.random.randn(n*m, target_rank) * 0.1
                U_new = U + perturbation
                new_error = compute_error(U_new, V, W)
                if new_error < current_error:
                    U = U_new
                    current_error = new_error
                
                perturbation = np.random.randn(m*p, target_rank) * 0.1
                V_new = V + perturbation
                new_error = compute_error(U, V_new, W)
                if new_error < current_error:
                    V = V_new
                    current_error = new_error
                
                perturbation = np.random.randn(p*n, target_rank) * 0.1
                W_new = W + perturbation
                new_error = compute_error(U, V, W_new)
                if new_error < current_error:
                    W = W_new
                    current_error = new_error
            
            if current_error < 1e-6:
                success = True
                break
        
        if success:
            best_rank = target_rank
            best_U = U
            best_V = V
            best_W = W
            break  # Found a lower rank, stop searching
    
    # If we found a better rank, use it; otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
