import time
import numpy as np
import itertools

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Decomposition (Rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # Index mapping:
            # U row: i*m + j
            # V row: j*p + k
            # W row: k*n + i (Trace convention W[k*n+i, r])
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def check_exact(U, V, W, rank):
        """
        Checks if U, V, W are integer or half-integer and satisfy the identity exactly.
        Identity: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
        """
        # Scale by 2 to check for integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        # Check if close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-5):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-5):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-5):
            return False
        
        # Convert to exact integers
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)
        
        # Verify identity with integers: sum_r (U_int/2)(V_int/2)(W_int/2) = delta
        # => sum_r U_int V_int W_int = 4 * delta
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i*m:(i+1)*m, :]
                V_slice = V_int[k*p:(k+1)*p, :]
                w_val = W_int[k*n + i]
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True

    # 2. Combinatorial Search for Rank Reduction
    # We attempt to find a rank-59 decomposition.
    # Strategy: Take the schoolbook decomposition (rank 60).
    # We look for a linear combination of 2 terms that can replace 3 terms, or similar.
    # A simpler robust approach for this specific size:
    # Randomly search for integer solutions to the tensor equation with rank 59.
    # However, pure random search in Z^59 is too large.
    
    # Instead, we use a "Greedy Merging" or "Strassen-like" construction.
    # For 2x5x6, we can view it as 2 blocks of 2x3x6? No, dimensions are fixed.
    # Let's try a local search around the schoolbook solution.
    # We will try to remove 1 term (rank 59) by adjusting the remaining 59 terms.
    # This is an underdetermined system of linear equations if we fix 1 term to be zero? No, we drop a term.
    # Let's drop term 't' and see if we can adjust the other terms to compensate.
    # The change in tensor T is sum_{r != t} delta_U[r] delta_V[r] delta_W[r] + ... 
    # This is non-linear.
    
    # Alternative: Use the known fact that rank(2x2x2) <= 7.
    # Can we embed a 2x2x2 Strassen decomposition into 2x5x6?
    # The 2x5x6 multiplication can be seen as 5x6 blocks of 2x1? No.
    # It is 2 rows of output, each is a sum of 5 products of 1x6 and 6x1? No.
    # Output C[i, k] = sum_j A[i, j] B[j, k].
    # i in {0,1}, k in {0..5}, j in {0..4}.
    
    # Let's try a randomized integer/half-integer search for rank 59.
    # We will use a heuristic: 
    # 1. Start with schoolbook.
    # 2. Pick a random term to "kill" (set to 0 in U, V, W for that column).
    # 3. The residual error is E.
    # 4. Try to absorb E into the other terms by small integer adjustments.
    
    # Given the time limit, a more reliable structural improvement is to use 
    # a pre-computed or algorithmically generated low-rank decomposition if possible.
    # Since I cannot rely on external libraries for specific tensor decompositions, 
    # I will implement a simple local search that tries to find a rank-59 solution 
    # by perturbing the schoolbook solution.
    
    # If we can't find 59 quickly, we return 60.
    # The inspiration says 59 is possible.
    
    # Let's try to construct a rank-59 solution explicitly using a "Strassen-like" trick on a subset.
    # Consider the sub-problem for i=0, k=0. C[0,0] = A[0,0]B[0,0] + ... + A[0,4]B[4,0].
    # This doesn't naturally form a 2x2x2 block unless we group indices.
    
    # Let's try a different approach: 
    # The tensor is T[i, j, k] = delta_{j, j'} (wait, no).
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j',k} in C_{i,k}.
    # Actually, the tensor M has indices i, j, k where j is the summation index.
    # M[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Yes, it's the standard matrix multiplication tensor.
    
    # We will attempt a random search for rank 59 with integer entries in {-1, 0, 1}.
    # This is a constraint satisfaction problem.
    
    # To stay within time limits, we will limit the search to a few hundred restarts.
    # If no solution is found, we return the schoolbook rank 60.
    
    # However, the prompt asks for a strategy that answers the open question about 'error_repair'.
    # The 'error_repair' likely refers to the ability to fix a nearly-correct solution.
    # My strategy: Start with a rank-59 "broken" solution (e.g., schoolbook with one term removed and random noise) and use a local repair search (gradient descent on integers or local search) to fix it.
    
    # Let's implement a local search for rank 59.
    # Variables: U, V, W of size (10, 59), (30, 59), (12, 59).
    # We restrict entries to {-1, 0, 1} to make it tractable.
    
    def try_find_rank_59():
        if time.time() - start_time > DEADLINE - 10:
            return None
            
        # We will use a simple simulated annealing or hill climbing on the integer grid.
        # Objective: Minimize || T_computed - T_target ||^2
        # T_target is the matrix multiplication tensor.
        
        # Represent the tensor as a 3D array T[i, j, k]
        # T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k]?
        # No, the tensor decomposition is:
        # C[i, k] = sum_r U[i, r] V[r, k] ? No, dimensions are different.
        
        # Standard Tensor Decomposition for MatMul (n x m) * (m x p):
        # C[i, k] = sum_{j=0}^{m-1} A[i, j] B[j, k]
        # This is a linear map from A, B to C.
        # The tensor M has indices i, j, k.
        # M[i, j, k] = 1. (Wait, is it 1 for all i, j, k? No.)
        # The multiplication is: C_{i,k} = \sum_j A_{i,j} B_{j,k}.
        # The tensor of structure coefficients is T_{i,j,k} such that C_{i,k} = \sum_{j} T_{i,j,k} A_{i,j} B_{j,k}.
        # Here T_{i,j,k} = 1 for all valid i, j, k.
        # Wait, the rank of this tensor is what we want to minimize.
        # A rank-1 decomposition is U[i, r] V[j, r] W[k, r]?
        # No, the standard decomposition is:
        # C_{i,k} = \sum_r U_{i,r} V_{r,k} is for (n x p) * (p x p)? No.
        
        # Let's stick to the interface definition:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # Identity: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
        
        # This identity is equivalent to:
        # For each i, k:
        # sum_r U[i, j1, r] V[j2, k, r] W[i, k, r] = delta_{j1, j2}
        # (Reshaping U to (n, m, rank), V to (m, p, rank), W to (n, p, rank))
        
        # Let's verify the schoolbook solution with this identity.
        # U[i, j, r] = 1 if r = i*m + j + k? No, r is the column index.
        # In schoolbook:
        # U[i*m+j, col] = 1
        # V[j*p+k, col] = 1
        # W[k*n+i, col] = 1
        # where col corresponds to the triple (i, j, k).
        # Let col = idx(i, j, k).
        # Sum over r:
        # Only r = idx(i, j1, k) contributes to U[i, j1, r].
        # Only r = idx(i, j2, k) contributes to V[j2, k, r]?
        # Wait, V index is j*p+k.
        # If r = idx(i, j1, k), then U[i, j1, r]=1.
        # For the product to be non-zero, we need V[j2, k, r] != 0 and W[i, k, r] != 0.
        # V[j2, k, r] is non-zero only if r corresponds to (i', j2, k') such that j2*p+k' = j2*p+k?
        # In schoolbook, V[j*p+k, col] = 1 where col = idx(i, j, k).
        # So V[j2, k, r] is 1 if r = idx(i, j2, k).
        # So for the product U[i, j1, r] V[j2, k, r] to be non-zero, we must have:
        # r = idx(i, j1, k) AND r = idx(i, j2, k).
        # This implies j1 = j2.
        # If j1 = j2, then r = idx(i, j1, k).
        # Then W[i, k, r] = W[k*n+i, idx(i, j1, k)].
        # In schoolbook, W[k*n+i, col] = 1 where col = idx(i, j, k).
        # So W[i, k, r] = 1.
        # Result: 1 * 1 * 1 = 1 = delta(j1, j2).
        # If j1 != j2, result is 0.
        # So the identity holds.
        
        # Now, we search for a rank-59 solution.
        # We will use a random restart local search.
        # We initialize U, V, W with random integers from {-1, 0, 1}.
        # We compute the residual R = T_computed - T_target.
        # We pick a random element in U, V, or W and flip it or change it by +/- 1.
        # If the residual norm decreases, we accept.
        
        # To speed up, we precompute the contribution of each term.
        
        def compute_residual_norm(U, V, W, rank):
            # Returns sum of squares of residuals
            # Residual for i, k, j1, j2 is:
            # sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] - delta(j1, j2)
            
            # We only need to check the diagonal (j1=j2) for 1 and off-diagonal for 0.
            # Actually, we can just compute the matrix M_{j1, j2} for each i, k.
            # M = sum_r U_i[:, r] * W[i, k, r] * V_k[:, r]^T ?
            # Let's reshape:
            # U_reshaped: (n, m, rank)
            # V_reshaped: (m, p, rank) -> (p, m, rank) to match V[j, k, r]
            # W_reshaped: (n, p, rank)
            
            # M_{j1, j2} for fixed i, k is:
            # sum_r U[i, j1, r] * W[i, k, r] * V[j2, k, r]
            
            # Let's vectorize this.
            # For each i, k:
            # vec = W[i, k, :]  # (rank,)
            # U_slice = U[i, :, :] # (m, rank)
            # V_slice = V[:, k, :] # (m, rank)
            # M = (U_slice * vec) @ V_slice.T # (m, m)
            
            total_err = 0.0
            for i in range(n):
                for k in range(p):
                    U_slice = U[i*m:(i+1)*m, :]
                    V_slice = V[k*p:(k+1)*p, :]
                    w_val = W[k*n + i]
                    
                    prod = U_slice * w_val[None, :]
                    mat = prod @ V_slice.T
                    target = np.eye(m)
                    diff = mat - target
                    total_err += np.sum(diff**2)
            return total_err

        # Local Search
        best_local_err = float('inf')
        best_local_U = None
        best_local_V = None
        best_local_W = None
        
        max_restarts = 50
        max_steps = 200
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Initialize with random integers {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, 59))
            V = np.random.choice([-1, 0, 1], size=(m*p, 59))
            W = np.random.choice([-1, 0, 1], size=(p*n, 59))
            
            # Sometimes start close to schoolbook (with one term removed)
            if restart % 2 == 0:
                U = best_U[:, :59].copy()
                V = best_V[:, :59].copy()
                W = best_W[:, :59].copy()
                # Add some noise
                U[np.random.choice(U.size, 10, replace=False)] += 1
                V[np.random.choice(V.size, 10, replace=False)] += 1
                W[np.random.choice(W.size, 10, replace=False)] += 1
                U = np.clip(U, -2, 2)
                V = np.clip(V, -2, 2)
                W = np.clip(W, -2, 2)
            
            current_err = compute_residual_norm(U, V, W, 59)
            
            if current_err < best_local_err:
                best_local_err = current_err
                best_local_U = U.copy()
                best_local_V = V.copy()
                best_local_W = W.copy()
                
            if current_err < 1e-5:
                # Found a solution!
                if check_exact(U, V, W, 59):
                    return U, V, W
            
            # Local search steps
            for step in range(max_steps):
                if time.time() - start_time > DEADLINE - 5:
                    break
                    
                # Pick a random matrix and a random element
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    idx = np.random.randint(n*m)
                    col = np.random.randint(59)
                    delta = np.random.choice([-1, 1])
                    old_val = U[idx, col]
                    U[idx, col] += delta
                    new_err = compute_residual_norm(U, V, W, 59)
                    if new_err >= current_err:
                        U[idx, col] = old_val
                    else:
                        current_err = new_err
                        if current_err < best_local_err:
                            best_local_err = current_err
                            best_local_U = U.copy()
                            best_local_V = V.copy()
                            best_local_W = W.copy()
                            if current_err < 1e-5:
                                if check_exact(U, V, W, 59):
                                    return U, V, W
                elif mat_choice == 1:
                    idx = np.random.randint(m*p)
                    col = np.random.randint(59)
                    delta = np.random.choice([-1, 1])
                    old_val = V[idx, col]
                    V[idx, col] += delta
                    new_err = compute_residual_norm(U, V, W, 59)
                    if new_err >= current_err:
                        V[idx, col] = old_val
                    else:
                        current_err = new_err
                        if current_err < best_local_err:
                            best_local_err = current_err
                            best_local_U = U.copy()
                            best_local_V = V.copy()
                            best_local_W = W.copy()
                            if current_err < 1e-5:
                                if check_exact(U, V, W, 59):
                                    return U, V, W
                else:
                    idx = np.random.randint(p*n)
                    col = np.random.randint(59)
                    delta = np.random.choice([-1, 1])
                    old_val = W[idx, col]
                    W[idx, col] += delta
                    new_err = compute_residual_norm(U, V, W, 59)
                    if new_err >= current_err:
                        W[idx, col] = old_val
                    else:
                        current_err = new_err
                        if current_err < best_local_err:
                            best_local_err = current_err
                            best_local_U = U.copy()
                            best_local_V = V.copy()
                            best_local_W = W.copy()
                            if current_err < 1e-5:
                                if check_exact(U, V, W, 59):
                                    return U, V, W
                            
        # If we found a very good local minimum, check it
        if best_local_U is not None and best_local_err < 1e-5:
             if check_exact(best_local_U, best_local_V, best_local_W, 59):
                 return best_local_U, best_local_V, best_local_W
                 
        return None

    # Attempt to find rank 59
    result_59 = try_find_rank_59()
    if result_59 is not None:
        best_rank = 59
        best_U, best_V, best_W = result_59

    # If we didn't find 59, we stick with 60.
    # We could try 58, but it's less likely to succeed in the time budget.
    
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

# EVOLVE-BLOCK-END
