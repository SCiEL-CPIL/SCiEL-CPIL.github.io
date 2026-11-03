import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Construct the trivial decomposition (Rank 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    # 2. Precompute target tensor
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
                valid_indices.append((a, b, c))
                
    def is_valid_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition is exact (within tolerance)."""
        for a, b, c in valid_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integer."""
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    # 3. Strassen 3x3x3 decomposition (rank 7)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5-M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7
    
    # For 3x3, we need to extend this. Let's use a block approach.
    # Actually, let's try a different approach: use the fact that we can share computations.
    
    # Strategy: For each pair of columns in B, we can potentially share Strassen-like computations.
    # But let's first try a simpler approach: use rank-7 Strassen for each "block" and see if we can merge.
    
    # Alternative: Direct search with better initialization using structure
    
    def als_step(U, V, W, T, R, max_iter=200, tol=1e-10):
        """ALS iterations."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for r in range(R):
                temp = np.einsum('abc,b->ac', T, V[:, r])
                U[:, r] = np.einsum('ac,c->a', temp, W[:, r])
                
            # Update V
            for r in range(R):
                temp = np.einsum('abc,a->bc', T, U[:, r])
                V[:, r] = np.einsum('bc,c->b', temp, W[:, r])
                
            # Update W
            for r in range(R):
                temp = np.einsum('abc,a->bc', T, U[:, r])
                W[:, r] = np.einsum('bc,b->c', temp, V[:, r])
                
            # Check convergence every 10 steps
            if it % 10 == 0:
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < tol:
                    break
                if prev_err - err < 1e-13 and it > 30:
                    break
                prev_err = err
                    
        return U, V, W

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # 4. Search for lower ranks with structured initialization
    # Key insight: The tensor has structure. Let's initialize using partial Strassen patterns.
    
    # Try ranks from 15 to 30
    rank_sequence = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    for R in rank_sequence:
        if time.time() > deadline - 15:
            break
            
        num_trials = 6 if R <= 20 else 4
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
                
            seed = trial * 10000 + R * 100 + 42
            np.random.seed(seed)
            
            # Initialize with structured pattern
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # For each rank component, assign a pattern
            for r in range(R):
                # Determine which (i,j,k) triple this rank component handles
                # Use a systematic assignment
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                
                # Set up the basic pattern
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                
                # Add some structure to allow sharing
                if r < R - 1:
                    # Add small perturbations to allow optimization
                    U[i * m + j, r] += np.random.randn() * 0.1
                    V[j * p + k, r] += np.random.randn() * 0.1
                    W[k * n + i, r] += np.random.randn() * 0.1
            
            # Run ALS
            U, V, W = als_step(U, V, W, target_tensor, R, max_iter=200)
            
            # Calculate error
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_approx - target_tensor)
            
            # If error is small, try polishing and rounding
            if error < 1e-3:
                # Try rounding to half-integers
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-8):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break
                
                # Also try without rounding
                if is_valid_decomposition(U, V, W, tol=1e-8):
                    # Try to find half-integer equivalent
                    U_r, V_r, W_r = round_to_half_integers(U, V, W)
                    if is_valid_decomposition(U_r, V_r, W_r, tol=1e-8):
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        best_rank = R
                        break
                    else:
                        # Use the exact solution
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = R
                        break
    
    # 5. Return the best result
    if best_rank < trivial_rank:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U_trivial),
            "V": to_list(V_trivial),
            "W": to_list(W_trivial)
        }
