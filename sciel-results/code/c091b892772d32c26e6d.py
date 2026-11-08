import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # We will try to find a solution with rank R < 45.
    # Start with schoolbook R=45.
    R_max = n * m * p
    
    # Try ranks from R_max - 1 downwards.
    # We can use a simple heuristic: try to zero out some columns and adjust others.
    # Or use a randomized search.
    
    # Since the system is bilinear, we can fix U and V and solve for W? No, W is also variable.
    # But if we fix U and V, the equations are linear in W?
    # sum_r U_ij_r V_jk_r W_ki_r = 1.
    # For fixed U, V, this is a linear system in W?
    # No, W appears multiplied by U and V.
    # Let's group by k, i.
    # For each k, i, we have sum_j (sum_r U_ij_r V_jk_r) W_ki_r = 1.
    # This is a system of linear equations in W_ki_r?
    # Let M_{ji, ki} = sum_r U_ij_r V_jk_r.
    # Then sum_j M_{ji, ki} W_ki_r ... wait.
    # The sum is over r.
    # sum_r (sum_j U_ij_r V_jk_r) W_ki_r = 1.
    # Let A_{ij, jk} = sum_r U_ij_r V_jk_r.
    # Then sum_r A_{ij, jk} W_ki_r = 1.
    # This is not linear in W because W is indexed by r, and A depends on r.
    # Actually, for each r, we have a term.
    # The equation is:
    # sum_{r=1}^R U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
    
    # If we fix U and V, the equations are linear in the variables W_{k,i,r}?
    # No, the product U*V*W is trilinear.
    # If we fix U and V, the term for a specific (i,j,k) is:
    # sum_r (U_{i,j,r} V_{j,k,r}) W_{k,i,r} = 1.
    # This is a linear equation in the variables W_{k,i,r} for a fixed (i,j,k)?
    # No, the coefficient of W_{k,i,r} is U_{i,j,r} V_{j,k,r}.
    # So for a fixed (i,j,k), we have:
    # sum_r (U_{i,j,r} V_{j,k,r}) W_{k,i,r} = 1.
    # This is a linear equation in the vector W_{k,i, :} (length R).
    # But we have 45 equations, and W has 15*R variables.
    # If we fix U and V, we have a linear system for W.
    # If the system has a solution, we can find W.
    # Then we can check if W can be made integer/half-integer.
    
    # Strategy:
    # 1. Randomly initialize U and V with small integers.
    # 2. Solve the linear system for W.
    # 3. Check if W is close to half-integers.
    # 4. If so, round and verify.
    
    # To improve the chance, we can start with the schoolbook solution and perturb it.
    
    # Schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # col = i*m*p + j*p + k.
    
    # Let's try to reduce rank by merging columns.
    # If two columns are identical in U and V, they can be merged in W?
    # If U_col1 == U_col2 and V_col1 == V_col2, then W_col1 + W_col2 works.
    
    # We will try to find a rank R solution by random search.
    
    best_rank = R_max
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U_sb = np.zeros((n*m, R_max))
    V_sb = np.zeros((m*p, R_max))
    W_sb = np.zeros((p*n, R_max))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to reduce rank
    for R in range(R_max - 1, 15, -1): # Try down to 15
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this R
        found = False
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V with random small integers
            # Or start with schoolbook and zero out some columns
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            
            # Copy first R columns of schoolbook?
            # This might not work well if the dependencies are complex.
            # Let's try random integers in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            
            # Solve for W
            # Equations: for each i,j,k
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system for W.
            # Let's build the matrix A for W.
            # Variables are W[0..p*n-1, 0..R-1]. Total (p*n)*R variables.
            # Number of equations: n*m*p = 45.
            # If R is small, we have more variables than equations, so solutions exist.
            # We want a solution where W is close to half-integers.
            
            # Let's use least squares to find a W that minimizes error.
            # But we want EXACT solution.
            # Since we have more variables than equations, we can find a solution in the null space.
            # Let's construct the system A * W_vec = b.
            # b is a vector of ones (length 45).
            # A is 45 x (p*n*R).
            
            A = np.zeros((n*m*p, p*n*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i*m*p + j*p + k
                        for r in range(R):
                            var_idx = (k*n + i) * R + r
                            A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                            
            b = np.ones(n*m*p)
            
            # Solve A * W_vec = b
            # Use least squares, but check for exact solution.
            # Since we are over reals, we can find a minimum norm solution.
            # Then check if it's close to half-integers.
            
            try:
                # Use numpy linalg.lstsq
                W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                error = np.linalg.norm(A @ W_vec - b)
                if error < 1e-9:
                    # Check if W_vec is close to half-integers
                    W_check = W_vec.reshape(p*n, R)
                    # Scale by 2 to check for integers
                    W_scaled = W_check * 2
                    W_rounded = np.round(W_scaled)
                    if np.allclose(W_scaled, W_rounded, atol=1e-6):
                        W_final = W_rounded / 2.0
                        # Verify
                        U_np = U
                        V_np = V
                        # Compute tensor
                        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                        # Check if all 1
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0
                                    for r in range(R):
                                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_final[k*n+i, r]
                                    if abs(val - 1) > 1e-9:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            best_rank = R
                            best_U = U
                            best_V = V
                            best_W = W_final
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            break
            
    # If we found a better rank, return it.
    # Otherwise, return schoolbook.
    
    # Convert to lists as required
    U_out = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_U]
    V_out = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_V]
    W_out = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
