import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank = n*m*p = 24)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    start_time = time.time()
    deadline = 140.0  # 10s margin

    # Search for lower rank using fast numpy solve + rounding
    # We iterate rank r from 23 down to 15
    for r in range(23, 14, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # More trials for lower ranks? Or just fixed?
        # Let's do 2000 trials per rank, but break early if time is up.
        max_trials = 2000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Random integer U, V
            # Use small integers to keep numbers manageable
            # -1, 0, 1
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)])
            V = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)])
            
            # Build linear system A * w = b for W
            # w is a vector of size p*n*r
            # Index mapping for w: idx = (k * n + i) * r + c
            num_vars = p * n * r
            num_eqs = n * m * p
            
            # A is num_eqs x num_vars
            # We can build A efficiently
            # A[eq_idx, w_base + c] = U[i*m+j][c] * V[j*p+k][c]
            # eq_idx = i * m * p + j * p + k
            
            # Instead of building full A, we can solve using least squares if overdetermined?
            # No, it's underdetermined usually (num_vars > num_eqs).
            # We want a specific solution. We can use np.linalg.lstsq to get minimum norm solution,
            # then check if it's close to half-integers.
            
            # Build A and b
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    u_row = U[i * m + j]
                    for k in range(p):
                        v_val = V[j * p + k]
                        w_base = (k * n + i) * r
                        # A[eq_idx, w_base : w_base+r] = u_row * v_val
                        A[eq_idx, w_base : w_base+r] = u_row * v_val
                        eq_idx += 1
            
            # Solve A w = b
            # Since it's underdetermined, lstsq gives min-norm solution.
            try:
                w_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # w_sol is a vector of floats
            # Check if it's close to half-integers
            # Multiply by 2 to check for integers
            w_scaled = w_sol * 2
            w_rounded = np.round(w_scaled)
            
            # Check if the difference is small
            if np.all(np.abs(w_scaled - w_rounded) < 1e-6):
                # Potential candidate
                # Convert w_rounded back to Fractions for exact check
                # w_rounded are integers (representing 2*W)
                # W_val = w_rounded / 2
                
                # Verify exactly
                valid = True
                # Reconstruct W from w_rounded
                # w_rounded is flattened W*2
                # W[k*n+i][c] = w_rounded[(k*n+i)*r + c] / 2
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            w_base = (k * n + i) * r
                            for c in range(r):
                                u_val = U[i * m + j][c]
                                v_val = V[j * p + k][c]
                                w_val_num = w_rounded[w_base + c]
                                # Product is u * v * (w_num / 2)
                                # Sum should be 1
                                # 2 * sum(u*v*w_num) should be 2
                                s += u_val * v_val * w_val_num
                            if s != 2:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        
                if valid:
                    # Found a solution!
                    # Convert to lists of floats (or ints)
                    U_out = U.tolist()
                    V_out = V.tolist()
                    W_out = []
                    for idx in range(p * n):
                        row = []
                        for c in range(r):
                            val = w_rounded[idx * r + c] / 2.0
                            # Ensure it's float
                            row.append(val)
                        W_out.append(row)
                    
                    return {"U": U_out, "V": V_out, "W": W_out}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
