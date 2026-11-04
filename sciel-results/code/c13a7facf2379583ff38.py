"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor as a 3D array: T[i,j,k] where indices are (i*m+j, j'*p+k, k'*n+i')
    # T[(i,j),(j',k),(k',i')] = 1 if j==j' and k==k' and i==i' else 0
    # We'll represent this as a function for checking
    
    def tensor_value(i, j, jp, k, kp, i2):
        """Return T value for indices: A index (i,j), B index (jp,k), C index (kp,i2)"""
        # B index is (j_prime, k) where j_prime = jp // p, k = jp % p... 
        # Wait, let me reclarify. V is indexed by (j, k) stored as j*p + k.
        # So for V index v_idx, j = v_idx // p, k = v_idx % p.
        # For W index w_idx, k = w_idx // n, i = w_idx % n.
        # The condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta for the multiplication
        # Actually the tensor is: for each (i,j) in A, (j',k) in B, (k',i') in C:
        # T = 1 if j==j' and k==k' and i==i'
        
        # Let me just check: does this correspond to a valid multiplication term?
        # A[i,j] * B[j',k] contributes to C[i,k] only if j==j'
        # So the tensor T_{(i,j),(j',k),(k',i')} = 1 iff j==j' and k==k' and i==i'
        
        # But in our parameterization:
        # U index: i*m + j (i in [0,n), j in [0,m))
        # V index: j'*p + k (j' in [0,m), k in [0,p))  
        # W index: k'*n + i' (k' in [0,p), i' in [0,n))
        
        # Condition: j == j' and k == k' and i == i'
        pass
    
    def is_valid(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        # For all i,j,j',k,k',i':
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j==j' and k==k' and i==i' else 0
        # This is O(n*m*m*p*p*n * rank) which is 3*4*4*4*4*3*rank = 2304*rank
        # For rank 48: 110592 operations, fine.
        
        # More efficiently: for each c, the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We can check by building the full tensor and comparing.
        
        # Build tensor T of shape (nm, mp, pn)
        T = np.zeros((nm, mp, pn))
        for c in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Target tensor
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m + j, j*p + k, k*n + i] = 1.0
        
        return np.allclose(T, T_target, atol=1e-9)
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_sb
    
    # Try to find lower rank
    # Strategy: Try ranks from best_rank-1 down, using optimization
    # For each rank r, try to find U,V,W satisfying the equations
    
    # Use a simpler approach: try to find a decomposition with rank r by
    # solving a system. For small rank reductions, we can try to find linear combinations.
    
    # Alternative: Use the fact that we can try random initializations and use
    # gradient descent to minimize the error.
    
    def try_find_decomposition(target_rank, max_iters=5000, seed=42):
        """Try to find a valid decomposition with target_rank using random init + gradient descent."""
        if target_rank >= best_rank:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random small values
        U = rng.randn(nm, target_rank) * 0.5
        V = rng.randn(mp, target_rank) * 0.5
        W = rng.randn(pn, target_rank) * 0.5
        
        # Target tensor
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m + j, j*p + k, k*n + i] = 1.0
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute current tensor
            T = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = T - T_target
            
            # Gradients
            dU = np.einsum('ijk,jc,kc->ic', err, V, W)
            dV = np.einsum('ijk,ic,kc->jc', err, U, W)
            dW = np.einsum('ijk,ic,jc->kc', err, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if np.max(np.abs(err)) < 1e-10:
                # Verify exactly
                if is_valid(U, V, W, target_rank):
                    return (U, V, W)
        
        # Final check
        T = np.einsum('ic,jc,kc->ijk', U, V, W)
        if np.max(np.abs(T - T_target)) < 1e-10:
            if is_valid(U, V, W, target_rank):
                return (U, V, W)
        
        return None
    
    # Try to find lower rank, starting from best_rank - 1
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Continue trying for even lower rank
            continue
        else:
            # If we can't find rank r, try a few more seeds
            for seed in range(1, 5):
                if time.time() > deadline - 10:
                    break
                result = try_find_decomposition(r, seed=seed)
                if result is not None:
                    U, V, W = result
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            if best_rank == r:
                # Found rank r, continue to r-1
                continue
            else:
                # Couldn't find rank r, but maybe lower ranks are possible?
                # Unlikely if r failed, but let's try a couple more
                break
    
    # Convert to lists with integer/half-integer entries if possible
    # The problem says entries should be integer/half-integer
    # Let's round to nearest half-integer and check validity
    
    def round_to_half(x):
        return np.round(x * 2) / 2.0
    
    U_r = round_to_half(np.array(best_result["U"]))
    V_r = round_to_half(np.array(best_result["V"]))
    W_r = round_to_half(np.array(best_result["W"]))
    
    if is_valid(U_r, V_r, W_r, best_rank):
        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    else:
        # Try rounding to integers
        U_r = np.round(np.array(best_result["U"]))
        V_r = np.round(np.array(best_result["V"]))
        W_r = np.round(np.array(best_result["W"]))
        if is_valid(U_r, V_r, W_r, best_rank):
            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
