import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # Schoolbook fallback
    def schoolbook():
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

    # Try to find lower rank
    # Target ranks: 22, 23, ..., 35
    for target_rank in range(22, n * m * p):
        if time.time() > deadline:
            break
        
        # Number of restarts depends on time remaining
        restarts = 100
        
        for _ in range(restarts):
            if time.time() > deadline:
                break
            
            # Initialize U and V with random half-integers
            # Using small integers to keep W potentially small
            u_vals = [-1, -0.5, 0, 0.5, 1]
            v_vals = [-1, -0.5, 0, 0.5, 1]
            
            U_rand = np.array([[random.choice(u_vals) for _ in range(target_rank)] for _ in range(n * m)])
            V_rand = np.array([[random.choice(v_vals) for _ in range(target_rank)] for _ in range(m * p)])
            
            # Construct linear system A * w = b
            # w is vector of W values, size p*n*target_rank
            # b is vector of 1s, size n*m*p
            num_eqs = n * m * p
            num_vars = p * n * target_rank
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(target_rank):
                            # Coefficient of W[k, i, r] is U[i, j, r] * V[j, k, r]
                            coeff = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                            if coeff != 0:
                                # Variable index for W[k, i, r]
                                # W is flattened as [k*n + i] for each r?
                                # Interface: W[k*n+i, r].
                                # So W is a matrix (p*n, rank).
                                # Vectorization: w[r * (p*n) + k*n + i]
                                var_idx = r * (p * n) + k * n + i
                                A[eq_idx, var_idx] = coeff
            
            # Solve least squares
            try:
                w_sol, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if solution is close to half-integers
            # Round to nearest half-integer
            w_rounded = np.round(w_sol * 2) / 2.0
            
            # Verify exact match with rounded W
            # Reconstruct T
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            # W[k, i, r]
                            w_val = w_rounded[r * (p * n) + k * n + i]
                            u_val = U_rand[i * m + j, r]
                            v_val = V_rand[j * p + k, r]
                            val += u_val * v_val * w_val
                        T_recon[i, j, k] = val
                        
            # Check if T_recon is all 1s
            if np.allclose(T_recon, 1.0, atol=1e-9):
                # Convert to lists of half-integers
                U_list = [[U_rand[i * m + j, r] for r in range(target_rank)] for i in range(n) for j in range(m)]
                V_list = [[V_rand[j * p + k, r] for r in range(target_rank)] for j in range(m) for k in range(p)]
                W_list = [[w_rounded[r * (p * n) + k * n + i] for r in range(target_rank)] for k in range(p) for i in range(n)]
                
                # Final verification
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(target_rank):
                                s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                            if abs(s - 1.0) > 1e-9:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    return {"U": U_list, "V": V_list, "W": W_list}

    return schoolbook()
# EVOLVE-BLOCK-END
