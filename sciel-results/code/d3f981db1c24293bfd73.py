import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    n, m, p = 2, 2, 8
    
    # Get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 15:
            break
        result = _search_low_rank(target_rank, remaining - 5)
        if result is not None:
            return result
    
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, nonzero_indices, num_a, num_b, num_c, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    # Compute T_hat[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficiently: for each r, outer product U[:,r] @ V[:,r] gives (num_a, num_b)
    # Then T_hat = sum_r W[c,r] * (U[:,r] @ V[:,r]) for each c
    
    # Build T_hat as (num_a, num_b, num_c)
    T_hat = np.zeros((num_a, num_b, num_c), dtype=float)
    for r in range(rank):
        UV_r = np.outer(U[:, r], V[:, r])  # (num_a, num_b)
        T_hat[:, :, r] = UV_r * 0  # placeholder, will use einsum
    # Use einsum for efficiency
    T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check non-zero entries
    for a, b, c in nonzero_indices:
        if abs(T_hat[a, b, c] - 1) > tol:
            return False
    
    # Check zero entries
    nonzero_set = set(nonzero_indices)
    for a in range(num_a):
        for b in range(num_b):
            for c in range(num_c):
                if (a, b, c) not in nonzero_set:
                    if abs(T_hat[a, b, c]) > tol:
                        return False
    
    return True

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the list of (a,b,c) indices where T=1
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
    
    nonzero_set = set(nonzero_indices)
    
    # Strategy: Start from Strassen-28 and try to reduce rank by finding
    # linear dependencies or using structured random search with refinement
    
    # First, try to find a rank-24 solution by exploiting block structure
    # The key idea: A is 2x2, so we can write A = [a00 a01; a10 a11]
    # B is 2x8, C = A*B is 2x8
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So C[0,k] = a00*B[0,k] + a01*B[1,k]
    #    C[1,k] = a10*B[0,k] + a11*B[1,k]
    
    # This means for each column k, we're doing a 2x2 times 2x1 multiplication
    # But the 2x2 matrix A is the same for all k!
    
    # So we can use a decomposition that shares the A-factors across all k.
    # Let's think of it as: for each of the 7 Strassen products, we need
    # to apply it to each of the 8 columns of B, but the A part is shared.
    
    # Actually, let's try a different approach: use the fact that the tensor
    # T[(i,j),(j',k),(i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
    # can be decomposed using the structure of the 2x2 multiplication.
    
    # Let's try random search with gradient refinement
    max_restarts = 500
    best_result = None
    
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart * 7919 + 12345)
        
        # Initialize with structured random values
        # Use small integers to increase chance of exact solution
        scale = rng.choice([0.5, 1, 2])
        
        U = rng.integers(-2, 3, size=(num_a, target_rank)).astype(float) * scale
        V = rng.integers(-2, 3, size=(num_b, target_rank)).astype(float) * scale
        W = rng.integers(-2, 3, size=(num_c, target_rank)).astype(float) * scale
        
        # Quick check on nonzero entries
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        max_err_nz = 0
        for a, b, c in nonzero_indices:
            err = abs(T_hat[a, b, c] - 1)
            if err > max_err_nz:
                max_err_nz = err
                if max_err_nz > 1.0:
                    break
        
        if max_err_nz < 0.5:
            # Check zero entries
            max_err_z = 0
            for a in range(num_a):
                for b in range(num_b):
                    for c in range(num_c):
                        if (a, b, c) not in nonzero_set:
                            err = abs(T_hat[a, b, c])
                            if err > max_err_z:
                                max_err_z = err
                                if max_err_z > 1.0:
                                    break
                    if max_err_z > 1.0:
                        break
                if max_err_z > 1.0:
                    break
            
            if max_err_z < 0.5:
                # Try to refine with gradient descent
                U_ref, V_ref, W_ref = _refine(U, V, W, nonzero_indices, nonzero_set, 
                                               num_a, num_b, num_c, time_budget, start)
                if U_ref is not None:
                    if _verify_decomposition(U_ref, V_ref, W_ref, nonzero_indices, 
                                             num_a, num_b, num_c, tol=1e-6):
                        U_ref = np.round(U_ref * 2) / 2
                        V_ref = np.round(V_ref * 2) / 2
                        W_ref = np.round(W_ref * 2) / 2
                        if _verify_decomposition(U_ref, V_ref, W_ref, nonzero_indices, 
                                                 num_a, num_b, num_c, tol=1e-6):
                            return {"U": U_ref.tolist(), "V": V_ref.tolist(), 
                                    "W": W_ref.tolist()}
    
    # Try structured approach: build from 2x2 Strassen with shared factors
    result = _structured_search(target_rank, time_budget, start)
    if result is not None:
        return result
    
    return None

def _refine(U, V, W, nonzero_indices, nonzero_set, num_a, num_b, num_c, 
            time_budget, start):
    """Refine a near-solution using gradient descent."""
    max_iters = 100
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() - start > time_budget - 5:
            break
        
        # Compute gradient
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Gradient w.r.t. U[a,r] = sum_{b,c} V[b,r]*W[c,r]*(T_hat[a,b,c] - T[a,b,c])
        # where T[a,b,c] = 1 if (a,b,c) in nonzero_indices, else 0
        
        # For efficiency, compute only on relevant entries
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for a, b, c in nonzero_indices:
            target = 1
            actual = T_hat[a, b, c]
            diff = actual - target
            for r in range(U.shape[1]):
                grad_U[a, r] += V[b, r] * W[c, r] * diff
                grad_V[b, r] += U[a, r] * W[c, r] * diff
                grad_W[c, r] += U[a, r] * V[b, r] * diff
        
        # Also add gradient from zero entries (sample a few)
        rng = np.random.default_rng(seed=it)
        num_samples = 100
        for _ in range(num_samples):
            a = rng.integers(0, num_a)
            b = rng.integers(0, num_b)
            c = rng.integers(0, num_c)
            if (a, b, c) in nonzero_set:
                continue
            target = 0
            actual = T_hat[a, b, c]
            diff = actual - target
            for r in range(U.shape[1]):
                grad_U[a, r] += V[b, r] * W[c, r] * diff
                grad_V[b, r] += U[a, r] * W[c, r] * diff
                grad_W[c, r] += U[a, r] * V[b, r] * diff
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        T_hat_new = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = 0
        for a, b, c in nonzero_indices:
            err = abs(T_hat_new[a, b, c] - 1)
            if err > max_err:
                max_err = err
        for a in range(num_a):
            for b in range(num_b):
                for c in range(num_c):
                    if (a, b, c) not in nonzero_set:
                        err = abs(T_hat_new[a, b, c])
                        if err > max_err:
                            max_err = err
        
        if max_err < 1e-6:
            return U, V, W
    
    return None, None, None

def _structured_search(target_rank, time_budget, start):
    """Structured search exploiting the 2x2 matrix multiplication structure."""
    # The key insight: A is 2x2, so we have 4 parameters for A.
    # B is 2x8, so 16 parameters. C = AB is 2x8, so 16 outputs.
    # The multiplication tensor has rank at least 16 (rank of the linear map B -> AB).
    
    # Let's try to build a decomposition where the U factors are shared across
    # the 8 columns of B, since A is the same for all columns.
    
    # For each of the 7 Strassen products, we have:
    # P_r = (linear combo of A entries) * (linear combo of B entries)
    # For column k of B, the B part involves B[0,k] and B[1,k]
    
    # So we can write:
    # C[i,k] = sum_r W[i,k,r] * U[r] * V[k,r]
    # where U[r] is a linear combo of A entries (same for all k),
    # V[k,r] is a linear combo of B[0,k] and B[1,k],
    # W[i,k,r] gives the contribution to C[i,k].
    
    # This suggests a decomposition where:
    # U has shape (4, R) - same for all k
    # V has shape (16, R) - depends on k
    # W has shape (16, R) - depends on k
    
    # Let's try R = 16, 18, 20, 22, 24
    
    # For each k (0..7), we're doing a 2x1 times 1x... no, it's 2x2 times 2x1
    # Actually for each column k, C[:,k] = A * B[:,k]
    # This is a 2x2 times 2x1 multiplication, which has tensor rank 7 (Strassen).
    # But A is shared!
    
    # Let's try a decomposition with rank 16:
    # For each k, use 2 terms (one for each row of A)
    # C[0,k] = a00*B[0,k] + a01*B[1,k]
    # C[1,k] = a10*B[0,k] + a11*B[1,k]
    # This gives rank 4*8/2 = 16? No, it's rank 16 with:
    # For k=0..7, r=2k: U = [1,0,0,0], V = [B[0,k]], W = [1,0]
    # For k=0..7, r=2k+1: U = [0,1,0,0], V = [B[1,k]], W = [0,1]
    # But this doesn't work because V needs to be linear in B entries.
    
    # Actually, the trivial decomposition has rank 4*8 = 32:
    # For each i,j,k: U[i,j,r] = delta, V[j,k,r] = delta, W[i,k,r] = delta
    # r indexes (i,j,k) with 2*2*8 = 32 terms.
    
    # Strassen gives 28 by sharing within each 2x2 block.
    
    # Let's try to find a rank-24 solution by searching over structured forms.
    
    # Use the fact that we can permute and scale the rank-1 terms.
    # Start from Strassen-28 and try to find 4 linear dependencies.
    
    U28 = np.array(_strassen_28()["U"])
    V28 = np.array(_strassen_28()["V"])
    W28 = np.array(_strassen_28()["W"])
    
    # Try to find a rank-24 sub-decomposition by removing 4 columns
    # and checking if the remaining 24 still span the tensor.
    
    # This is a combinatorial problem, so let's try random subsets
    rng = np.random.default_rng(seed=42)
    
    for attempt in range(1000):
        if time.time() - start > time_budget:
            break
        
        # Try removing 4 columns
        cols_to_remove = rng.choice(28, size=4, replace=False)
        cols_to_keep = [c for c in range(28) if c not in cols_to_remove]
        
        U_try = U28[:, cols_to_keep]
        V_try = V28[:, cols_to_keep]
        W_try = W28[:, cols_to_keep]
        
        if _verify_decomposition(U_try, V_try, W_try, 
                                  [(i*2+j, j*8+k, k*2+i) for i in range(2) for j in range(2) for k in range(8)],
                                  4, 16, 16, tol=1e-6):
            return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    return None
