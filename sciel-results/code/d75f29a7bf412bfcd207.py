import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Lower bound
    lb = max(n*m, m*p, p*n)
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook solution as fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank
    # We know lb=28. Try R from lb to best_rank-1
    for R in range(lb, best_rank):
        if time.time() > deadline:
            break
            
        attempts = 0
        max_attempts = 50  # Limit attempts per R
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Randomize U and V
            # Use small integers to increase chance of rational/integer W
            # Values from {-1, 0, 1}
            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            
            # Ensure some non-zero entries
            for r in range(R):
                # Randomly select indices
                u_idx = np.random.choice(n*m, size=max(1, n*m//4), replace=False)
                v_idx = np.random.choice(m*p, size=max(1, m*p//4), replace=False)
                
                for idx in u_idx:
                    U[idx, r] = np.random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, r] = np.random.choice([-1, 1])
            
            # Build linear system for W
            # Equations: for each i, j, k: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
            # Unknowns: W[k*n+i, r]
            # Total unknowns: p*n*R
            num_unknowns = p * n * R
            num_equations = n * m * p
            
            A = np.zeros((num_equations, num_unknowns), dtype=float)
            b = np.ones(num_equations, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                        for r in range(R):
                            coeff = U[i*m+j, r] * V[j*p+k, r]
                            if coeff != 0:
                                # Column index for W[k*n+i, r]
                                # W is indexed by [k*n+i, r]
                                # Flatten index: (k*n+i) * R + r
                                col_idx = (k*n+i) * R + r
                                A[eq_idx, col_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Reshape W
            W = W_flat.reshape(p*n, R)
            
            # Check if W is close to half-integers
            # Multiply by 2 to check for integers
            W_2 = W * 2
            W_rounded = np.round(W_2)
            
            # Check error
            err = np.linalg.norm(W_2 - W_rounded)
            if err < 1e-6:
                # Verify exactness with rounded values
                W_test = W_rounded / 2.0
                # Reconstruct tensor
                # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W_test[k*n+i, r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Convert to lists of floats (or ints if integer)
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_test.tolist()
                    
                    # Clean up floats that are integers
                    def clean_list(l):
                        out = []
                        for row in l:
                            new_row = []
                            for val in row:
                                if abs(val - round(val)) < 1e-9:
                                    new_row.append(int(round(val)))
                                else:
                                    new_row.append(val)
                            out.append(new_row)
                        return out
                    
                    return {
                        "U": clean_list(U_list),
                        "V": clean_list(V_list),
                        "W": clean_list(W_list)
                    }
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
