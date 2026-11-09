import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook as fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using least squares and rounding
    # We try ranks from best_rank - 1 down to a minimum
    min_rank_to_try = 20
    
    # Pre-compute the tensor indices
    # T[i, j, k] = 1 for all i, j, k
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We fix U and V to be random sparse 0/1 matrices and solve for W?
    # No, the equation is multilinear.
    
    # Alternative: Fix U and V, and treat the equation as linear in W?
    # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
    # This is NOT linear in W because the coefficient of W depends on j which is summed.
    # Wait.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Decomposition: C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) * (sum_k V[j, k, r] B[j, k])? No.
    
    # The standard decomposition is:
    # C[i, k] = sum_r M_r(i, k)
    # M_r(i, k) = (sum_j U[i, j, r] A[i, j]) * (sum_k V[j, k, r] B[j, k])? No.
    
    # The tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Here U[i, j, r] is the coefficient of A[i, j].
    # V[j, k, r] is the coefficient of B[j, k].
    # W[k, i, r] is the coefficient of C[k, i]? No, C[i, k].
    
    # So:
    # 1 = sum_r U[i, m+j, r] * V[j, p+k, r] * W[k, n+i, r]
    
    # This is a system of non-linear equations.
    
    # However, if we fix U and V, we can write:
    # sum_r W[k, n+i, r] * (U[i, m+j, r] * V[j, p+k, r]) = 1
    # This is still not linear in W because the term in parenthesis depends on j.
    # But for a fixed (i, k), the equation is:
    # sum_j sum_r W[k, n+i, r] * U[i, m+j, r] * V[j, p+k, r] = 1
    # Let Y[i, k, r] = W[k, n+i, r].
    # Then: sum_j sum_r Y[i, k, r] * U[i, m+j, r] * V[j, p+k, r] = 1
    # This is linear in Y (which is W) IF U and V are fixed.
    
    # So the strategy is:
    # 1. Fix U and V to be random sparse matrices with entries in {0, 1, -1}.
    # 2. Solve the linear system for W.
    # 3. Round W to nearest half-integer.
    # 4. Verify.
    
    np.random.seed(42)
    
    for rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Initialize U and V with random sparse values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            # Randomly set some entries to 1 or -1
            # Density should be high enough to be solvable
            for idx in range(n * m):
                for r in range(rank):
                    U[idx, r] = np.random.choice([-1, 1])
                        
            for idx in range(m * p):
                for r in range(rank):
                    V[idx, r] = np.random.choice([-1, 1])
            
            # Construct the linear system for W
            # Variables: W[k, n+i, r] for all k, i, r
            # Total variables: p * n * rank
            num_vars = p * n * rank
            
            # Equations: for all i, j, k
            # sum_r W[k, n+i, r] * U[i, m+j, r] * V[j, p+k, r] = 1
            # Note: The sum is over r.
            # For a fixed (i, k), we have:
            # sum_j sum_r W[k, n+i, r] * U[i, m+j, r] * V[j, p+k, r] = 1
            # This gives one equation for each (i, k).
            # Total equations: n * p
            
            # Wait, the tensor condition is for each (i, j, k).
            # T[i, j, k] = 1.
            # The equation is:
            # sum_r U[i, m+j, r] * V[j, p+k, r] * W[k, n+i, r] = 1
            # This must hold for ALL i, j, k.
            
            # Let's group by (i, k).
            # For fixed i, k, the equation is:
            # sum_j [ sum_r U[i, m+j, r] * V[j, p+k, r] * W[k, n+i, r] ] = 1
            # This is NOT linear in W because W is inside the sum over j?
            # No, W is W[k, n+i, r]. The index k and i are fixed.
            # The sum is over j and r.
            # Let A_jr = U[i, m+j, r] * V[j, p+k, r].
            # Then sum_j sum_r A_jr * W[k, n+i, r] = 1.
            # This IS linear in W.
            # The coefficient of W[k, n+i, r] is sum_j A_jr = sum_j U[i, m+j, r] * V[j, p+k, r].
            
            # So we have a system:
            # For each (i, k):
            # sum_r ( sum_j U[i, m+j, r] * V[j, p+k, r] ) * W[k, n+i, r] = 1
            
            # Let M[i, k, r] = sum_j U[i, m+j, r] * V[j, p+k, r]
            # Then: sum_r M[i, k, r] * W[k, n+i, r] = 1
            
            # This is a system of n*p equations with p*n*rank variables.
            # Since n*p = p*n, we have n*p equations and n*p*rank variables.
            # This is under-determined if rank > 1.
            # We can find a minimum-norm solution or a sparse solution.
            
            # Construct matrix A_sys (size n*p x p*n*rank)
            # Row index: idx_ik = i * p + k
            # Col index: idx_kir = (k * n + i) * rank + r
            
            A_sys = np.zeros((n * p, p * n * rank))
            b_sys = np.ones(n * p)
            
            for i in range(n):
                for k in range(p):
                    row_idx = i * p + k
                    for r in range(rank):
                        # Calculate M[i, k, r]
                        M_ikr = 0
                        for j in range(m):
                            M_ikr += U[i * m + j, r] * V[j * p + k, r]
                        
                        col_idx = (k * n + i) * rank + r
                        A_sys[row_idx, col_idx] = M_ikr
            
            # Solve A_sys * W_flat = b_sys
            # Use least squares
            try:
                W_flat, residuals, rank_sys, s = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            except:
                continue
                
            # Reshape W_flat to W
            W = W_flat.reshape((p * n, rank))
            
            # Round to nearest half-integer
            W_rounded = np.round(W * 2) / 2.0
            
            # Verify the solution
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s_val = 0
                        for r in range(rank):
                            s_val += U[i * m + j, r] * V[j * p + k, r] * W_rounded[k * n + i, r]
                        if abs(s_val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                # Convert to lists of floats
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_rounded.tolist()
                
                # Check if entries are half-integers (they should be due to rounding)
                best_rank = rank
                best_result = {
                    "U": U_list,
                    "V": V_list,
                    "W": W_list
                }
                break # Found a solution for this rank, try lower rank
            
    return best_result
