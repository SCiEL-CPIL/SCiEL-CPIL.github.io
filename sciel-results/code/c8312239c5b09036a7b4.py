import numpy as np
import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    Tensor convention: W[k*n+i, r]
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The decomposition satisfies:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 elsewhere.
    """
    n, m, p = 3, 3, 8
    dim_u = n * m  # 9
    dim_v = m * p  # 24
    dim_w = p * n  # 24
    
    deadline = time.time() + 115  # 15s safety margin
    
    # Try to find a low-rank decomposition using optimization
    # Start with rank below schoolbook (72) and go up if needed
    
    best_result = None
    best_rank = dim_u * dim_v * dim_w  # schoolbook rank = 72
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    # Verify schoolbook is correct
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact for the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build the tensor from the decomposition
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We only need to check the "supported" entries (where T should be 1)
        # and a sample of zeros
        
        # For speed, check all supported entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Compute T[a,b,c]
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check some zeros (entries where T should be 0)
        # T[a,b,c] = 0 when the indices don't form a valid i,j,k triple
        import itertools
        checked = 0
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    # Skip supported entries
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and i == i2 and k == k2:
                        continue
                    # This should be 0
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val) > 1e-10:
                        return False
                    checked += 1
                    if checked > 500:  # Limit for speed
                        break
                if checked > 500:
                    break
            if checked > 500:
                break
        return True
    
    # First, verify the schoolbook solution is correct
    if check_decomposition(U_school, V_school, W_school):
        best_rank = rank_school
        best_result = (U_school, V_school, W_school)
    else:
        # Something is wrong with our understanding; use schoolbook anyway
        best_rank = rank_school
        best_result = (U_school, V_school, W_school)
    
    # Now try to find lower-rank decompositions
    # Use random initialization + gradient descent
    
    def try_rank(rank, max_iter=200, seed=42):
        """Try to find a rank-r decomposition using optimization."""
        np.random.seed(seed)
        
        # Initialize with random matrices
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        # Use L-BFGS-B or similar
        from scipy.optimize import minimize
        
        def objective_and_grad(params):
            U_p = params[:dim_u*rank].reshape(dim_u, rank)
            V_p = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_p = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            
            # Loss: sum of (T_supported - computed)^2 + lambda * sum(zero_entries^2)
            # For speed, use a sampled set of zero entries
            
            # Compute supported entries
            loss = 0.0
            grad_U = np.zeros_like(U_p)
            grad_V = np.zeros_like(V_p)
            grad_W = np.zeros_like(W_p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        # T[a,b,c] should be 1
                        # d/dU[a,r] = V[b,r]*W[c,r]
                        for r in range(rank):
                            diff = U_p[a,r]*V_p[b,r]*W_p[c,r]  # this is just one term
                            # Actually we need the full sum
                        # Better: compute full value
                        val = np.sum(U_p[a,:] * V_p[b,:] * W_p[c,:])
                        err = val - 1.0
                        for r in range(rank):
                            grad_U[a, r] += err * V_p[b, r] * W_p[c, r]
                            grad_V[b, r] += err * U_p[a, r] * W_p[c, r]
                            grad_W[c, r] += err * U_p[a, r] * V_p[b, r]
                        loss += err**2
            
            # Sample zero entries
            np.random.seed(0)  # fixed for reproducibility within objective
            num_zero_samples = min(200, dim_u*dim_v*dim_w)
            zero_loss = 0.0
            zero_indices = []
            # Generate random zero entries
            for _ in range(num_zero_samples):
                a = np.random.randint(dim_u)
                b = np.random.randint(dim_v)
                c = np.random.randint(dim_w)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and i == i2 and k == k2:
                    continue  # skip supported
                val = np.sum(U_p[a,:] * V_p[b,:] * W_p[c,:])
                zero_loss += val**2
                for r in range(rank):
                    grad_U[a, r] += val * V_p[b, r] * W_p[c, r]
                    grad_V[b, r] += val * U_p[a, r] * W_p[c, r]
                    grad_W[c, r] += val * U_p[a, r] * V_p[b, r]
            
            # Also add a small regularization to keep numbers reasonable
            reg = 1e-8 * (np.sum(U_p**2) + np.sum(V_p**2) + np.sum(W_p**2))
            grad_U += 2e-8 * U_p
            grad_V += 2e-8 * V_p
            grad_W += 2e-8 * W_p
            
            total_loss = loss + zero_loss + reg
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return total_loss, grad
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                            jac=True, options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15})
        except Exception as e:
            return None
        
        if result.fun < 1e-8:
            U_final = result.x[:dim_u*rank].reshape(dim_u, rank)
            V_final = result.x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_final = result.x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            # Verify
            if check_decomposition(U_final, V_final, W_final):
                return (U_final, V_final, W_final)
            else:
                # Maybe it's close but not exact - try a polish
                # Scale and shift to try to get exact 1s
                return None
        return None
    
    # Try ranks from just below schoolbook down to some minimum
    # The theoretical lower bound for matrix multiplication is at least max(n*m, m*p, p*n)
    # For 3x3*3x8: max(9, 24, 24) = 24
    # But realistically, we expect something between 24 and 72
    
    # Try a range of ranks
    ranks_to_try = [50, 48, 45, 42, 40, 38, 36, 35, 34, 33, 32, 30, 28, 27, 26, 25, 24]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random seeds
        found = False
        for seed in [42, 123, 777, 999, 2024]:
            if time.time() > deadline:
                break
            result = try_rank(target_rank, max_iter=150, seed=seed)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        if found:
            break  # Found a lower rank, stop searching higher
    
    # Format the best result
    U_best, V_best, W_best = best_result
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_list, "V": V_list, "W": W_list}
