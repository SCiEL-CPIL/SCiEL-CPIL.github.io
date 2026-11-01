import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Search for lower rank
    # Try ranks from 30 to best_rank - 1
    for R in range(30, best_rank):
        if time.time() > DEADLINE - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(2000):
            if time.time() > DEADLINE - 5:
                break
                
            # Randomly initialize V and W with small integers
            V = np.random.randint(-2, 3, size=(m * p, R))
            W = np.random.randint(-2, 3, size=(p * n, R))
            
            U = np.zeros((n * m, R))
            
            valid = True
            # Solve for U for each j
            for j in range(m):
                # For this j, we need to find U[i, j, t] for i in [0, n), t in [0, R)
                # Equation: sum_t U[i, j, t] * V[j, k, t] * W[k, i, t] = 1 for all i, k
                # Let's build the linear system M * u = b
                # u is vector of length n * R (flattened U[:, j, :])
                # M has shape (n * p, n * R)
                # Row index: (i, k) -> i * p + k
                # Col index: (i, t) -> i * R + t
                # Coefficient M[i*p+k, i*R+t] = V[j*p+k, t] * W[k*n+i, t]
                
                M = np.zeros((n * p, n * R))
                b = np.ones(n * p)
                
                for i in range(n):
                    for k in range(p):
                        row_idx = i * p + k
                        for t in range(R):
                            col_idx = i * R + t
                            M[row_idx, col_idx] = V[j * p + k, t] * W[k * n + i, t]
                
                # Solve M * u = b
                # Use least squares to find a solution
                try:
                    u, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Reshape u back to (n, R)
                u_matrix = u.reshape((n, R))
                
                # Try to find integer or half-integer solution
                # First try rounding to integers
                u_int = np.round(u_matrix).astype(int)
                
                # Check if u_int satisfies the equations exactly
                residual = 0
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for t in range(R):
                            val += u_int[i, t] * V[j * p + k, t] * W[k * n + i, t]
                        if val != 1:
                            residual += 1
                            break
                    if residual > 0:
                        break
                
                if residual == 0:
                    u_matrix = u_int.astype(float)
                else:
                    # Try half-integers
                    u_half = np.round(u_matrix * 2) / 2.0
                    residual_half = 0
                    for i in range(n):
                        for k in range(p):
                            val = 0
                            for t in range(R):
                                val += u_half[i, t] * V[j * p + k, t] * W[k * n + i, t]
                            if abs(val - 1) > 1e-9:
                                residual_half += 1
                                break
                        if residual_half > 0:
                            break
                    
                    if residual_half > 0:
                        # Try to find a better solution using local search
                        # Perturb the solution and try to round
                        found = False
                        for _ in range(10):
                            u_perturbed = u_matrix + np.random.normal(0, 0.1, size=u_matrix.shape)
                            u_rounded = np.round(u_perturbed * 2) / 2.0
                            residual_check = 0
                            for i in range(n):
                                for k in range(p):
                                    val = 0
                                    for t in range(R):
                                        val += u_rounded[i, t] * V[j * p + k, t] * W[k * n + i, t]
                                    if abs(val - 1) > 1e-9:
                                        residual_check += 1
                                        break
                                if residual_check > 0:
                                    break
                            if residual_check == 0:
                                u_matrix = u_rounded
                                found = True
                                break
                        
                        if not found:
                            valid = False
                            break
                    else:
                        u_matrix = u_half
                
                # Store U for this j
                for i in range(n):
                    U[i * m + j, :] = u_matrix[i, :]
            
            if valid:
                # Final verification of the whole tensor
                # Check sum_t U[i,j,t] V[j,k,t] W[k,i,t] == 1 for all i,j,k
                full_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for t in range(R):
                                val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                            if abs(val - 1) > 1e-9:
                                full_valid = False
                                break
                        if not full_valid:
                            break
                    if not full_valid:
                        break
                
                if full_valid:
                    # Convert to lists, ensuring integer or half-integer values
                    U_list = []
                    for row in U:
                        U_list.append([float(x) if x % 1 != 0 else int(x) for x in row])
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
