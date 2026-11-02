import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook

    # The theoretical lower bound is max(n*m, m*p, p*n) = 20, but that's for rectangular matrix multiplication in general.
    # For 3x4 * 4x5, we want to beat 60.
    # We will try to find a decomposition with rank < 60.
    # Strategy: 
    # 1. Random search over U, V with small integer entries.
    # 2. Solve for W using linear algebra.
    # 3. Check if W has integer/half-integer entries.
    
    # We can use a structured approach:
    # Let U and V be random matrices with entries in {-1, 0, 1}.
    # For each (i, k), the system is sum_j U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # Let u_i be the i-th block of U (size m x r).
    # Let v_k be the k-th block of V (size m x r).
    # Let w_{ik} be the (k,i)-th row of W (size 1 x r).
    # The equation is: sum_{j=0}^{m-1} sum_{c=0}^{r-1} u_i[j,c] v_k[j,c] w_{ik}[c] = 1.
    # This can be written as: <u_i * v_k, w_{ik}> = 1, where * is element-wise.
    # Let M_{ik} be a matrix of size m x r, where M_{ik}[j,c] = u_i[j,c] * v_k[j,c].
    # Then sum_j M_{ik}[j,c] * w_{ik}[c] = 1.
    # This is a linear system M_{ik}^T w_{ik} = 1 (vector of ones of size m).
    # Wait, the sum is over j and c.
    # Let's define vector z_{ik} of size m*r where z_{ik}[j,c] = u_i[j,c] * v_k[j,c].
    # And vector w_{ik}^* of size m*r where w_{ik}^*[j,c] = w_{ik}[c].
    # Then <z_{ik}, w_{ik}^*> = 1.
    # This means for a fixed (i,k), w_{ik} is a solution to a linear system.
    # Specifically, let A_{ik} be the m x r matrix with A_{ik}[j,c] = u_i[j,c] * v_k[j,c].
    # Then A_{ik} w_{ik} = 1 (m-vector).
    
    # If we fix U and V, we can solve for W.
    # We want W to have half-integer entries.
    
    # Let's try to find U, V such that the systems A_{ik} w_{ik} = 1 have solutions with half-integer entries.
    
    # We will search for ranks from 59 down to 50.
    for r in range(59, 49, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries in {-1, 0, 1}
            # Using {-1, 0, 1} increases the chance of linear dependencies.
            U = np.random.randint(-1, 2, (n * m, r))
            V = np.random.randint(-1, 2, (m * p, r))
            
            # To ensure solvability, we might need to adjust.
            # Let's check if the system is solvable and if the solution is "nice".
            
            W = np.zeros((p * n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build matrix A (m x r)
                    # A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        A[j, :] = u_row * v_row
                    
                    # Solve A * w = 1
                    # We want w to be half-integers.
                    # Let's use least squares first.
                    try:
                        # If A is rank deficient, lstsq gives the minimum norm solution.
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Round to nearest half-integer
                        w_rounded = np.round(w * 2) / 2
                        
                        # Check if it's a valid solution
                        if np.allclose(A @ w_rounded, np.ones(m), atol=1e-9):
                            # Check if entries are half-integers (they are by construction of rounding)
                            # But we should ensure they are not too large? The problem doesn't restrict magnitude, only type.
                            # However, very large numbers might cause precision issues.
                            W[k * n + i, :] = w_rounded
                        else:
                            # Try to find a solution in the null space?
                            # Or just fail this attempt.
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                # T_{ijk} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # We want T_{ijk} = 1 if it's the multiplication tensor.
                # The multiplication tensor has T_{ijk} = 1 for all i,j,k? 
                # Wait, the tensor for matrix multiplication C = AB is:
                # C_{ik} = sum_j A_{ij} B_{jk}
                # So the tensor T has indices (i, j, k) and T_{ijk} = 1 for all i, j, k?
                # No, T_{ijk} corresponds to the coefficient of A_{ij} B_{jk} in C_{ik}.
                # So T_{ijk} = 1 for all valid i, j, k.
                
                # Let's verify:
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(val - 1) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    # We found a valid decomposition with rank r
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Ensure half-integers are represented correctly
                    # The problem says "integer/half-integer entries".
                    # Numpy floats like 0.5 are fine.
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
