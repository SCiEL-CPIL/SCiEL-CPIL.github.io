import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Target tensor T_flat
    # T[i, j, k, l, m, n] is not quite right, let's use the code's definition
    # T[i, j, j, k, i, k] = 1.0
    # This is a 6-dimensional tensor, but effectively it's the matrix multiplication tensor.
    # We can represent it as a 3D tensor T[i, j, k] where the "multiplication" is implicit?
    # No, the decomposition is U (nm x R), V (mp x R), W (pn x R).
    # The identity is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) is a valid triple, else 0.
    # Wait, the tensor T is defined such that T_flat is the target.
    # Let's create the target tensor T of shape (n, m, m, p, n, p) as in the original code.
    
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
    T_flat = T.flatten()
    
    def verify_decomposition(U, V, W, R):
        """
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        Returns True if the decomposition is exact.
        """
        # Compute reconstruction
        # recon[i, j, j, k, i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can vectorize this check.
        # For each r, we have a rank-1 tensor.
        # Sum them up.
        
        # Reshape U, V, W to (R, n, m), (R, m, p), (R, p, n)
        U_reshaped = U.T.reshape((R, n, m))
        V_reshaped = V.T.reshape((R, m, p))
        W_reshaped = W.T.reshape((R, p, n))
        
        # Compute sum_r U_r[i, j] * V_r[j, k] * W_r[k, i]
        # This is a tensor of shape (n, m, p) ? No, the indices are (i, j, k) for the product.
        # But the target tensor has indices (i, j, j', k, i', k').
        # The non-zero entries are at (i, j, j, k, i, k).
        # So we just need to check if the scalar product matches for each (i, j, k).
        
        # Let's compute the reconstructed tensor R_recon of shape (n, m, p)
        # R_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can do this with einsum
        # U is (R, n, m), V is (R, m, p), W is (R, p, n)
        # We want sum_r U[r, i, j] * V[r, j, k] * W[r, k, i]
        
        # Note: W index is (k, i). In W_reshaped (R, p, n), index is [r, k, i].
        # So:
        recon = np.einsum('rij,rjk,rki->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Now compare with T
        # T has non-zeros at (i, j, k) where the original indices match.
        # In our T (n, m, m, p, n, p), non-zeros are at (i, j, j, k, i, k).
        # So for each (i, j, k), T[i, j, j, k, i, k] should equal recon[i, j, k].
        
        # Extract the relevant slice from T
        # T_slice[i, j, k] = T[i, j, j, k, i, k]
        # We can just loop or use advanced indexing.
        
        # Since n, m, p are small, a loop is fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if np.abs(recon[i, j, k] - T[i, j, j, k, i, k]) > 1e-9:
                        return False
        return True

    def schoolbook_decomposition():
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

    best_solution = None
    best_rank = n * m * p
    
    # Try to find a better rank using random search
    # We will try ranks from 20 to 83
    # For each rank, we will do a limited number of random attempts
    
    # To make random search effective, we restrict entries to {-1, -0.5, 0, 0.5, 1}
    # We will use a greedy approach: build the decomposition column by column.
    
    def try_greedy_rank(R, max_attempts=50):
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Residual tensor
            # We store the residual as a dictionary of (i, j, k) -> value
            # Initially, T has 1s at all (i, j, k)
            residual = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[(i, j, k)] = 1.0
            
            for r in range(R):
                if time.time() > deadline:
                    return None
                
                # Find a rank-1 tensor that best approximates the residual
                # We will try to find u, v, w with entries in {-1, -0.5, 0, 0.5, 1}
                # such that the dot product with the residual is maximized (or error minimized)
                
                # Simplified: Randomly sample u, v and solve for w?
                # Or just random sample u, v, w and check if it reduces the error.
                
                # Let's try to find a "good" rank-1 tensor.
                # We pick a random non-zero entry from the residual to anchor the tensor.
                if not residual:
                    break
                    
                keys = list(residual.keys())
                target_i, target_j, target_k = keys[np.random.randint(len(keys))]
                
                # We want u[target_i, target_j] * v[target_j, target_k] * w[target_k, target_i] to match residual[(target_i, target_j, target_k)]
                # Let's try to set u, v, w to have small supports.
                
                # Randomly generate u, v, w with small entries
                u = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n, m))
                v = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m, p))
                w = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p, n))
                
                # Check if this rank-1 tensor is "compatible" with the residual
                # i.e., it doesn't introduce large errors where residual is 0
                
                # Compute the error introduced
                # Actually, we just want to see if subtracting this tensor helps.
                # We will subtract it and see if the max absolute residual decreases.
                
                # But we must ensure the entries are integer/half-integer.
                
                # Let's just try this random tensor.
                # Update U, V, W
                U[:, r] = u.flatten()
                V[:, r] = v.flatten()
                W[:, r] = w.flatten()
                
                # Update residual
                # For each (i, j, k), residual[(i, j, k)] -= u[i, j] * v[j, k] * w[k, i]
                # We only care about keys that are in the residual or might become non-zero.
                # Since u, v, w are sparse-ish, we can iterate over non-zeros.
                
                # To be efficient, we just recompute the residual for all (i, j, k)
                # This is O(nmp) which is small (84).
                
                new_residual = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = residual.get((i, j, k), 0.0) - u[i, j] * v[j, k] * w[k, i]
                            if abs(val) > 1e-9:
                                new_residual[(i, j, k)] = val
                            # If val is close to 0, we drop it
                            
                # Check if the new residual is "smaller" than the old one
                # We can compare the sum of squares or max absolute value.
                # If it's not smaller, we might reject this step?
                # For now, we accept it and hope for the best.
                
                residual = new_residual
                
                if not residual:
                    # We have a perfect decomposition!
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
            # If we finished all R columns and residual is not empty, this attempt failed
            # But we can check if the residual is very small (numerical error)
            if residual:
                max_err = max(abs(v) for v in residual.values())
                if max_err < 1e-9:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    
        return None

    # Try ranks from 20 to 83
    # We will spend most of the time on lower ranks, but they are harder.
    # We will allocate time based on rank.
    
    # Actually, let's just iterate and stop when time runs out.
    for R in range(20, 84):
        if time.time() > deadline:
            break
            
        # Number of attempts depends on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Try a few attempts for this rank
        # Lower ranks are harder, so maybe fewer attempts?
        # Or more attempts?
        # Let's do 10 attempts per rank.
        
        sol = try_greedy_rank(R, max_attempts=10)
        if sol is not None:
            best_solution = sol
            best_rank = R
            break # Found a solution, stop searching for lower ranks? 
                   # No, we want the MINIMAL rank. 
                   # So we should continue to lower R? 
                   # But we are iterating from 20 up. 
                   # So the first one we find is the minimal one in this range.
                   # However, we might have missed a solution for R=19.
                   # We started at 20.
                   # Let's start from 1? No, 1 is too low.
                   # The theoretical lower bound is higher.
                   
    if best_solution is None:
        best_solution = schoolbook_decomposition()
        best_rank = n * m * p
        
    return best_solution
