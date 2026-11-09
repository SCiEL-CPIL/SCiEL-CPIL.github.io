import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    # U: (n*m) x r
    # V: (m*p) x r
    # W: (p*n) x r
    
    # Target rank to try
    # Schoolbook is n*m*p = 24
    # Try to find lower rank
    
    best_result = None
    best_rank = n * m * p
    
    # Initialize schoolbook as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to improve
    # We search for r in [18, 23]
    for r in range(18, 24):
        if time.time() > deadline:
            break
            
        # Number of restarts
        # We can do many quick restarts
        max_restarts = 500
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Generate random sparse 0/1 U and V
            # Density should be low to allow linear independence
            # Each column c of U and V defines a "term"
            # U[i*m+j, c] and V[j*p+k, c]
            
            # Strategy: Randomly assign non-zero entries to U and V
            # To make the linear system solvable, we need the columns of the coefficient matrix to be independent enough.
            
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            
            # Randomly place 1s
            # Ensure each column has at least one 1 in U and V
            # And that the pattern allows solving for W
            
            # Simple heuristic: 
            # For each column c, pick a random subset of indices for U and V
            # But they must be "compatible" to contribute to the tensor.
            
            # Actually, let's just randomize U and V with low probability
            # and solve.
            
            # Probability of 1
            p_u = 0.3
            p_v = 0.3
            
            U = (np.random.rand(n * m, r) < p_u).astype(int)
            V = (np.random.rand(m * p, r) < p_v).astype(int)
            
            # Construct the linear system A * W = b
            # Variables W are flattened (p*n) * r
            # Equations are for each i, j, k
            # Eq idx: i * m * p + j * p + k
            # Var idx: k * n * r + i * r + c  -> (k*n + i) * r + c
            
            num_eqs = n * m * p
            num_vars = p * n * r
            
            A = np.zeros((num_eqs, num_vars), dtype=float)
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # Coeff for W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c]
                        for c in range(r):
                            var_idx = (k * n + i) * r + c
                            coeff = U[i * m + j, c] * V[j * p + k, c]
                            A[eq_idx, var_idx] = coeff
            
            # Solve A * W_flat = b
            try:
                # Use least squares or direct solve if square/full rank
                # Since num_vars >= num_eqs usually, we use lstsq
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is exact (residuals close to 0)
                if residuals.size > 0 and residuals[0] > 1e-9:
                    continue
                    
                # Check if W entries are rational (close to fractions with small denom)
                # We can just check if they are close to 0, 0.5, 1, etc.
                # Or just keep them as floats? The task says "integer/half-integer entries".
                # So we must snap to 0.5 increments.
                
                W_rounded = np.round(W_flat * 2) / 2
                
                # Verify with rounded W
                W_check = np.zeros((p * n, r))
                W_check_flat = W_rounded.reshape(-1)
                
                # Reconstruct W matrix from flat
                # W[k*n+i, c] is at index (k*n+i)*r + c
                for c in range(r):
                    for kn_i in range(p * n):
                        W_check[kn_i, c] = W_rounded[kn_i * r + c]
                        
                # Verify tensor
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W_check[k * n + i, c]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        
                if valid:
                    # Convert to lists of ints/floats
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_check.tolist()
                    
                    # Clean up floats (e.g., 1.0 -> 1)
                    def clean_matrix(m):
                        return [[int(x) if x == int(x) else x for x in row] for row in m]
                    
                    best_result = {"U": clean_matrix(U_list), "V": clean_matrix(V_list), "W": clean_matrix(W_list)}
                    best_rank = r
                    found = True
                    break # Found for this r, break restarts
                    
            except np.linalg.LinAlgError:
                continue
                
        if found:
            # We found a decomposition of rank r.
            # We could try lower r, but the loop handles that.
            # Since we iterate r from 18 upwards, if we find one at 18, we break outer loop?
            # No, we want the minimal. So if we find one, we can break the r loop if we are searching upwards?
            # Wait, we search 18, 19, 20...
            # If we find one at 18, that's the best possible (assuming 18 is min).
            # So we can break.
            break

    return best_result
