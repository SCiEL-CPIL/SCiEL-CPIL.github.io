import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
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

    # Try to find a lower rank
    # We will try ranks 59, 58, ...
    # For each rank, we try a few random U, V
    # and solve for W
    
    # To solve for W, we use the equations:
    # sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta(j1, j2) * delta(i, i1) * delta(k, k2)
    # Wait, the LHS is the coefficient of A[i1, j1] B[j2, k2] in the expansion.
    # The RHS is the coefficient in AB.
    # (AB)[i, k] = sum_j A[i, j] B[j, k].
    # Coeff of A[i1, j1] B[j2, k2] is 1 if i=i1, k=k2, j1=j2. Else 0.
    
    # So for each (i1, j1, j2, k2), we have an equation for each (i, k)?
    # No, the equation is for the whole matrix C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # This must equal sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i, j]) * (sum_j V[j*p+k, r] B[j, k]).
    
    # Let's expand the RHS:
    # sum_r W[k*n+i, r] * (sum_j1 U[i*m+j1, r] A[i, j1]) * (sum_j2 V[j2*p+k, r] B[j2, k]).
    # = sum_j1 sum_j2 A[i, j1] B[j2, k] * (sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r]).
    
    # This must equal sum_j A[i, j] B[j, k] for all i, k.
    # So for each i, k, j1, j2:
    # Coeff of A[i, j1] B[j2, k] on RHS is sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r].
    # Coeff on LHS is delta(j1, j2).
    
    # So we have equations for each (i, k, j1, j2):
    # sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] = delta(j1, j2).
    
    # Number of equations: n * p * m * m = 2 * 6 * 5 * 5 = 300.
    # Unknowns: R * n * p = R * 12.
    
    # We can solve this linear system for W given U, V.
    
    for rank in range(59, 40, -1):
        if time.time() > deadline:
            break
            
        # Try a few random U, V
        for trial in range(10):
            if time.time() > deadline:
                break
                
            # Random U, V with small integers
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build the linear system A_w * W = b
            # W is flattened to (R * n * p)
            # Equation index: (i, k, j1, j2)
            # W index: (k * n + i) * rank + r? No, W is (p*n, rank).
            # W[k*n+i, r] is the variable.
            
            num_eqs = n * p * m * m
            num_vars = rank * n * p
            
            A_mat = np.zeros((num_eqs, num_vars), dtype=float)
            b_vec = np.zeros(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            # Target: delta(j1, j2)
                            b_vec[eq_idx] = 1.0 if j1 == j2 else 0.0
                            
                            # Coefficients for W[k*n+i, r]
                            for r in range(rank):
                                coeff = U[i * m + j1, r] * V[j2 * p + k, r]
                                # Variable is W[k*n+i, r]
                                var_idx = (k * n + i) * rank + r
                                A_mat[eq_idx, var_idx] = coeff
                            
                            eq_idx += 1
            
            # Solve A_mat * W_flat = b_vec
            # Use least squares
            W_flat, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check if the solution is exact (residuals close to 0)
            if residuals and residuals[0] > 1e-6:
                continue
                
            # Check if W_flat is half-integer
            W_check = W_flat.reshape(p * n, rank)
            valid = True
            for row in range(p * n):
                for col in range(rank):
                    val = W_check[row, col]
                    # Check if val is close to x.0 or x.5
                    if abs(val - round(val * 2) / 2.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists of floats/ints
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_check.tolist()
                # Snap to half-integers
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        W_list[i][j] = round(W_list[i][j] * 2) / 2.0
                        
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_school, "V": V_school, "W": W_school}
