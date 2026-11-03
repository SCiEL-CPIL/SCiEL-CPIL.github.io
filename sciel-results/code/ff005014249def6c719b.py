import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target rank to try
    target_rank = 84
    
    # Build the target tensor T (sparse representation)
    # T is a dictionary mapping (alpha, beta, gamma) to value
    # alpha in [0, nm), beta in [0, mp), gamma in [0, pn)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i for some i,j,k
    # Otherwise 0
    
    # We don't need to store the full tensor, just the non-zero entries
    non_zero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                non_zero.append((alpha, beta, gamma))
    
    # We will try to find U, V, W of rank target_rank
    # such that sum_t U[alpha, t] V[beta, t] W[gamma, t] == T[alpha, beta, gamma]
    
    # Start time
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    best_result = None
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        rank = target_rank
        # Initialize U, V, W with random values
        # Scale to avoid overflow
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # ALS iterations
        for iter in range(100):
            if time.time() > deadline:
                break
                
            # Update U
            # For each alpha, solve least squares
            # Minimize sum_{beta, gamma} (T[alpha, beta, gamma] - sum_t U[alpha, t] V[beta, t] W[gamma, t])^2
            # Let M[beta, gamma, t] = V[beta, t] W[gamma, t]
            # M_flat[ (beta*pn + gamma), t ]
            # Target[ (beta*pn + gamma) ] = T[alpha, beta, gamma]
            
            # Construct M_flat
            # V is (mp, rank), W is (pn, rank)
            # M_flat is (mp*pn, rank)
            # M_flat[idx, t] = V[beta, t] * W[gamma, t]
            # idx = beta * pn + gamma
            
            # We can compute this efficiently
            # V[:, :, None] * W[None, :, :] -> (mp, pn, rank)
            # Reshape to (mp*pn, rank)
            M_flat = (V[:, :, None] * W[None, :, :]).reshape(-1, rank)
            
            for alpha in range(nm):
                # Target vector for this alpha
                # T[alpha, beta, gamma] is 1 if (alpha, beta, gamma) in non_zero, else 0
                # We can create a dense vector of size mp*pn
                # But it's sparse.
                # However, mp*pn = 30*18 = 540. Small enough to be dense.
                target_vec = np.zeros(mp * pn)
                # Fill in the non-zero entries for this alpha
                # We need to iterate over non_zero and pick those with this alpha
                # This is slow if done inside the loop.
                # Pre-compute for each alpha the list of (beta, gamma)
                
                # Pre-computation outside the loop?
                # Let's pre-compute alpha_to_indices
                # alpha_to_indices[alpha] = list of (beta, gamma)
                
                # For now, just iterate
                for (a, b, g) in non_zero:
                    if a == alpha:
                        idx = b * pn + g
                        target_vec[idx] = 1.0
                
                # Solve M_flat @ U[alpha, :] = target_vec
                # U[alpha, :] is the unknown vector
                sol, residuals, rank_sol, sv = np.linalg.lstsq(M_flat, target_vec, rcond=None)
                U[alpha, :] = sol
            
            # Update V
            # Symmetric
            M_flat = (U[:, :, None] * W[None, :, :]).reshape(-1, rank)
            # Wait, U is (nm, rank), W is (pn, rank)
            # M_flat[idx, t] = U[alpha, t] * W[gamma, t]
            # idx = alpha * pn + gamma ? No.
            # The term is U[alpha, t] V[beta, t] W[gamma, t]
            # For fixed beta, we want to solve for V[beta, :]
            # Sum_{alpha, gamma} (T[alpha, beta, gamma] - sum_t U[alpha, t] V[beta, t] W[gamma, t])^2
            # Let M_flat[ (alpha*pn + gamma), t ] = U[alpha, t] * W[gamma, t]
            # Target[ (alpha*pn + gamma) ] = T[alpha, beta, gamma]
            
            # M_flat size: (nm*pn, rank)
            M_flat = (U[:, :, None] * W[None, :, :]).reshape(-1, rank)
            # Note: U is (nm, rank), W is (pn, rank)
            # U[:, :, None] is (nm, rank, 1)
            # W[None, :, :] is (1, pn, rank)
            # Product is (nm, pn, rank)
            # Reshape to (nm*pn, rank)
            # Index: alpha * pn + gamma
            
            for beta in range(mp):
                target_vec = np.zeros(nm * pn)
                for (a, b, g) in non_zero:
                    if b == beta:
                        idx = a * pn + g
                        target_vec[idx] = 1.0
                
                sol, residuals, rank_sol, sv = np.linalg.lstsq(M_flat, target_vec, rcond=None)
                V[beta, :] = sol
            
            # Update W
            # For fixed gamma, solve for W[gamma, :]
            # Sum_{alpha, beta} (T[alpha, beta, gamma] - sum_t U[alpha, t] V[beta, t] W[gamma, t])^2
            # M_flat[ (alpha*mp + beta), t ] = U[alpha, t] * V[beta, t]
            # Target[ (alpha*mp + beta) ] = T[alpha, beta, gamma]
            
            M_flat = (U[:, :, None] * V[None, :, :]).reshape(-1, rank)
            # U (nm, rank), V (mp, rank)
            # Product (nm, mp, rank)
            # Reshape (nm*mp, rank)
            # Index: alpha * mp + beta
            
            for gamma in range(pn):
                target_vec = np.zeros(nm * mp)
                for (a, b, g) in non_zero:
                    if g == gamma:
                        idx = a * mp + b
                        target_vec[idx] = 1.0
                
                sol, residuals, rank_sol, sv = np.linalg.lstsq(M_flat, target_vec, rcond=None)
                W[gamma, :] = sol
            
            # Check error
            # Calculate error on non-zero entries
            error = 0.0
            for (a, b, g) in non_zero:
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                error += (val - 1.0) ** 2
            
            # Also check some zero entries?
            # If error is small, check zero entries
            if error < 1e-6:
                # Check zero entries
                zero_error = 0.0
                # Sample some zero entries
                # To be safe, check all? 8100 entries.
                # 8100 * 84 ops. Fast.
                for a in range(nm):
                    for b in range(mp):
                        for g in range(pn):
                            if (a, b, g) not in non_zero:
                                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                                zero_error += val ** 2
                                if zero_error > 1e-6:
                                    break
                        if zero_error > 1e-6:
                            break
                    if zero_error > 1e-6:
                        break
                
                if zero_error < 1e-6:
                    # Found a solution!
                    # Check if entries are close to half-integers
                    # Multiply by 2 and round
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify exactness with rounded values
                    exact_error = 0.0
                    for (a, b, g) in non_zero:
                        val = np.sum(U_int[a, :] * V_int[b, :] * W_int[g, :])
                        exact_error += (val - 1.0) ** 2
                    
                    zero_error_int = 0.0
                    for a in range(nm):
                        for b in range(mp):
                            for g in range(pn):
                                if (a, b, g) not in non_zero:
                                    val = np.sum(U_int[a, :] * V_int[b, :] * W_int[g, :])
                                    zero_error_int += val ** 2
                                    if zero_error_int > 1e-9:
                                        break
                            if zero_error_int > 1e-9:
                                break
                        if zero_error_int > 1e-9:
                            break
                    
                    if exact_error < 1e-9 and zero_error_int < 1e-9:
                        # Convert to lists
                        best_result = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        return best_result
        
    # If no solution found, return seed
    return construct_seed()

def construct_seed():
    n, m, p = DIMENSIONS
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
