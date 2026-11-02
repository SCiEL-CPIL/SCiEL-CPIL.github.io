import numpy as np
import itertools
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Schoolbook rank
    rank_school = n * m * p  # 48
    
    target = np.ones((n, m, p))
    target_flat = target.flatten()
    
    def try_random_rank(rank, max_attempts, time_budget):
        """Try to find a valid decomposition at the given rank."""
        if time.time() > time_budget:
            return None
        
        U_best = None
        V_best = None
        W_best = None
        
        for attempt in range(max_attempts):
            if time.time() > time_budget:
                break
            
            # Use sparser random matrices to make the system more tractable
            # Density around 0.3-0.5 for better conditioning
            density = 0.4
            U = (np.random.random((n * m, rank)) < density).astype(float) * 2 - 1
            V = (np.random.random((m * p, rank)) < density).astype(float) * 2 - 1
            # Make entries exactly -1, 0, or 1
            U = np.where(U != 0, np.sign(U), 0)
            V = np.where(V != 0, np.sign(V), 0)
            
            # Build the constraint matrix A such that A @ W_flat = target_flat
            # The tensor decomposition says: target[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For each (i,j,k), we have: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # So A[row, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
            A = np.zeros((n * m * p, p * n * rank))
            
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i * m + j, r]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = V[j * p + k, r]
                            if v_val == 0:
                                continue
                            A[i * m * p + j * p + k, (k * n + i) * rank + r] = u_val * v_val
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integers
            W_int = np.round(W_flat)
            error = np.linalg.norm(A @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                if np.allclose(W_flat, W_int, atol=1e-8):
                    U_best = U
                    V_best = V
                    W_best = W
                    break
        
        if U_best is not None:
            U_list = [[int(U_best[i, r]) for r in range(rank)] for i in range(n * m)]
            V_list = [[int(V_best[i, r]) for r in range(rank)] for i in range(m * p)]
            W_list = [[int(W_best[i, r]) for r in range(rank)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_structured_perturbation(rank, time_budget):
        """
        Start from a valid rank-(rank+1) decomposition and try to merge two terms.
        We use the schoolbook decomposition as a starting point and try to find
        linear dependencies that allow rank reduction.
        """
        if time.time() > time_budget:
            return None
        
        # Start with a valid decomposition of rank n*m*p
        # and try to find linear combinations that cancel one term
        
        # Build the full schoolbook tensor representation
        # For each term r in range(n*m*p), define the index (i,j,k)
        terms = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
                    col += 1
        
        # The schoolbook has rank n*m*p. We want to find if any subset of rank+1 terms
        # is linearly dependent in a way that allows us to express one as a combination of others.
        
        # Try random subsets of size rank+1 and check if they span a rank < rank+1 subspace
        # If so, we can find coefficients to express the last one as a combination of the first rank.
        
        for attempt in range(200):
            if time.time() > time_budget:
                break
            
            # Pick rank+1 random terms from the schoolbook
            chosen_indices = np.random.choice(len(terms), size=rank+1, replace=False)
            chosen_terms = [terms[idx] for idx in chosen_indices]
            
            # For these rank+1 terms, build the constraint matrix
            # and check if the system is underdetermined (has non-trivial null space)
            
            A = np.zeros((n * m * p, (rank + 1) * p * n))
            
            for t_idx, (i, j, k) in enumerate(chosen_terms):
                # Term t contributes to target[i,j,k] via U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # For a rank-(rank+1) partial decomposition using just these terms:
                # We set U, V, W to be identity-like for the selected terms
                # Actually, let's think differently.
                
                # For each term (i,j,k), the contribution is to target[i,j,k] only.
                # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
                # So A[row, t * p*n + ... ] - this gets complicated.
                pass
        
        return None
    
    # Strategy 1: Try random search for ranks 47 down to 44
    for rank in [47, 46, 45, 44]:
        if time.time() > deadline - 10:
            break
        
        # Allocate time proportional to remaining budget
        remaining = deadline - 10 - time.time()
        if remaining < 2:
            break
        
        # More attempts for higher ranks (easier to find)
        if rank == 47:
            max_attempts = int(remaining * 50)  # ~50 per second
        elif rank == 46:
            max_attempts = int(remaining * 30)
        elif rank == 45:
            max_attempts = int(remaining * 15)
        else:
            max_attempts = int(remaining * 5)
        
        max_attempts = max(max_attempts, 50)
        
        result = try_random_rank(rank, max_attempts, deadline - 10)
        if result is not None:
            return result
    
    # Strategy 2: Try with different density patterns
    if time.time() < deadline - 15:
        for rank in [47, 46]:
            if time.time() > deadline - 10:
                break
            
            for density in [0.2, 0.3, 0.5, 0.6]:
                if time.time() > deadline - 10:
                    break
                
                max_attempts = int((deadline - 10 - time.time()) * 20)
                max_attempts = max(max_attempts, 30)
                
                # Override density in try_random_rank by using a helper
                U_best = None
                V_best = None
                W_best = None
                
                for attempt in range(max_attempts):
                    if time.time() > deadline - 10:
                        break
                    
                    U = (np.random.random((n * m, rank)) < density).astype(float) * 2 - 1
                    V = (np.random.random((m * p, rank)) < density).astype(float) * 2 - 1
                    U = np.where(U != 0, np.sign(U), 0)
                    V = np.where(V != 0, np.sign(V), 0)
                    
                    A = np.zeros((n * m * p, p * n * rank))
                    
                    for r in range(rank):
                        for i in range(n):
                            for j in range(m):
                                u_val = U[i * m + j, r]
                                if u_val == 0:
                                    continue
                                for k in range(p):
                                    v_val = V[j * p + k, r]
                                    if v_val == 0:
                                        continue
                                    A[i * m * p + j * p + k, (k * n + i) * rank + r] = u_val * v_val
                    
                    try:
                        W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, target_flat, rcond=None)
                    except:
                        continue
                    
                    W_int = np.round(W_flat)
                    error = np.linalg.norm(A @ W_int - target_flat)
                    
                    if error < 1e-9:
                        W = W_int.reshape(p * n, rank)
                        if np.allclose(W_flat, W_int, atol=1e-8):
                            U_best = U
                            V_best = V
                            W_best = W
                            break
                
                if U_best is not None:
                    U_list = [[int(U_best[i, r]) for r in range(rank)] for i in range(n * m)]
                    V_list = [[int(V_best[i, r]) for r in range(rank)] for i in range(m * p)]
                    W_list = [[int(W_best[i, r]) for r in range(rank)] for i in range(p * n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
