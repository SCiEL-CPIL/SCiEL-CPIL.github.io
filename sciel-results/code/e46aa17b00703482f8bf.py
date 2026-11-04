"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill

    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p  # 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p  # 72

    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        # Convert to numpy for speed
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Systematic search for rank reduction
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Key insight: For fixed (i,k), we have 3 equations (j=0,1,2):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let W_ik = W[k*n+i, c] (depends on c)
    # Then: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W_ik = 1
    
    # This is a linear system in the products. Let's try structured decompositions.
    
    # Approach: Try to find decompositions where each column c contributes
    # to a specific pattern of (i,j,k) indices.
    
    # For rank r, we can think of it as r terms in a sum.
    # Each term c has U_c (9-dim), V_c (24-dim), W_c (24-dim).
    # The constraint is that the outer product U_c ⊗ V_c ⊗ W_c, when 
    # contracted appropriately, sums to 1 over all (i,j,k).
    
    # Let's try a different approach: use the fact that we can factor
    # the all-ones tensor in various ways.
    
    # For example, if we let:
    # U[i*m+j, c] = a_c * b_{i,j,c}
    # V[j*p+k, c] = a_c * c_{j,k,c}  
    # W[k*n+i, c] = a_c * d_{k,i,c}
    # Then sum_c a_c^3 * b_{i,j,c} * c_{j,k,c} * d_{k,i,c} = 1
    
    # This is getting complex. Let's try a more direct numerical approach.
    
    # Try to solve using optimization or direct construction.
    
    # Alternative: Try specific structured patterns
    
    # Pattern 1: For each i, k, try to cover all j with shared structure
    # If we fix i and k, we need for each j:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try: for each (i,k), use 3 columns that only affect that (i,k) pair
    # and cover j=0,1,2. This gives 3*3*8 = 72 columns (schoolbook).
    
    # To reduce rank, we need columns that affect multiple (i,k) pairs.
    
    # Let's try: columns that affect 2 (i,k) pairs each.
    # If a column affects (i1,k1) and (i2,k2), it can contribute to both.
    
    # This is still complex. Let's try a numerical search with gradient-free methods.
    
    # Actually, let's try a smarter structured approach:
    
    # For rank r, initialize U, V, W with small random values, then
    # try to adjust them to satisfy the constraints.
    
    # Use a local search / hill climbing approach.
    
    def local_search(U, V, W, rank, max_iter=5000):
        """Try to improve a decomposition via local search."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        # Compute current error
        def compute_error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (val - 1.0) ** 2
            return err
        
        current_err = compute_error()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if current_err < 1e-12:
                break
            
            # Pick a random entry to perturb
            choice = np.random.randint(3)
            if choice == 0:
                idx = np.random.randint(n * m)
                col = np.random.randint(rank)
                delta = np.random.choice([-0.5, 0.5])
                old_val = U[idx, col]
                U[idx, col] += delta
                new_err = compute_error()
                if new_err < current_err:
                    current_err = new_err
                else:
                    U[idx, col] = old_val
            elif choice == 1:
                idx = np.random.randint(m * p)
                col = np.random.randint(rank)
                delta = np.random.choice([-0.5, 0.5])
                old_val = V[idx, col]
                V[idx, col] += delta
                new_err = compute_error()
                if new_err < current_err:
                    current_err = new_err
                else:
                    V[idx, col] = old_val
            else:
                idx = np.random.randint(p * n)
                col = np.random.randint(rank)
                delta = np.random.choice([-0.5, 0.5])
                old_val = W[idx, col]
                W[idx, col] += delta
                new_err = compute_error()
                if new_err < current_err:
                    current_err = new_err
                else:
                    W[idx, col] = old_val
        
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        return U, V, W, current_err
    
    # Try different target ranks with local search
    for target_rank in range(71, 40, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with small random values
            U = np.random.randint(-2, 3, (n * m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, (m * p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, (p * n, target_rank)) * 0.5
            
            U, V, W, err = local_search(U, V, W, target_rank, max_iter=2000)
            
            if err < 1e-9:
                # Verify exactly
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank < 71:
            break
    
    # If still at 72, try a more aggressive approach with larger perturbations
    if best_rank == 72 and time.time() < deadline - 10:
        for target_rank in range(70, 50, -1):
            if time.time() > deadline:
                break
            
            for attempt in range(2):
                if time.time() > deadline:
                    break
                
                np.random.seed(attempt * 5000 + target_rank)
                
                # Initialize with a pattern: start from schoolbook and try to merge columns
                U = np.zeros((n * m, target_rank), dtype=float)
                V = np.zeros((m * p, target_rank), dtype=float)
                W = np.zeros((p * n, target_rank), dtype=float)
                
                # Fill first target_rank columns with schoolbook-like patterns
                for c in range(target_rank):
                    i = c // (m * p)
                    rem = c % (m * p)
                    j = rem // p
                    k = rem % p
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                
                # Now try to adjust to make it work with fewer columns
                U, V, W, err = local_search(U, V, W, target_rank, max_iter=3000)
                
                if err < 1e-9:
                    if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
            
            if best_rank < 70:
                break
    
    return best_result
# EVOLVE-BLOCK-END
