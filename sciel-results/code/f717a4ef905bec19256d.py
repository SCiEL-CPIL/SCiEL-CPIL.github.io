"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 18
    
    # Schoolbook for fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for target_rank in [16, 17]:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        result = _search_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    return {"U": U_school, "V": V_school, "W": W_school}


def _search_rank(n, m, p, target_rank):
    """Search for a rank-target_rank decomposition of the all-ones tensor."""
    import random
    import sys
    random.seed(12345)
    
    nU = n * m  # 6
    nV = m * p  # 9
    nW = p * n  # 6
    
    # We'll use a structured approach:
    # Each rank-1 term c has supports: S_U_c subset of [n]x[m], S_V_c subset of [m]x[p], S_W_c subset of [p]x[n]
    # The contribution to (i,j,k) is non-zero only if (i,j) in S_U_c, (j,k) in S_V_c, (k,i) in S_W_c
    
    # For the sum to equal 1 for all (i,j,k), we need careful design.
    
    # Strategy: use a "block" structure inspired by Strassen.
    # For 2x2x2, Strassen uses 7 terms with specific support patterns.
    # For 2x3x3, we can try to embed similar patterns.
    
    # Let's try: each term covers a "box" of indices.
    # Term c: U has entries for i in I_c, j in J_c; V for j in J_c, k in K_c; W for k in K_c, i in I_c
    # where I_c subset of [n], J_c subset of [m], K_c subset of [p]
    
    # If we use full boxes: U[i,j,c]=a_c for i in I_c, j in J_c; etc.
    # Then contribution to (i,j,k) is a_c*b_c*d_c if i in I_c, j in J_c, k in K_c
    
    # This is a set cover + scaling problem.
    
    # Simpler: try random sparse patterns and solve linear system.
    # Fix: for each term c, set one entry in U, V, W to 1, solve for others.
    
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return None
        
        # Generate random supports
        U_sup = []
        V_sup = []
        W_sup = []
        
        for c in range(target_rank):
            # Random supports of size 1-3
            us = random.randint(1, 3)
            vs = random.randint(1, 3)
            ws = random.randint(1, 3)
            
            U_sup.append(random.sample([(i,j) for i in range(n) for j in range(m)], min(us, n*m)))
            V_sup.append(random.sample([(j,k) for j in range(m) for k in range(p)], min(vs, m*p)))
            W_sup.append(random.sample([(k,i) for k in range(p) for i in range(n)], min(ws, p*n)))
        
        # Build linear system
        # Unknowns: for each c, for each (i,j) in U_sup[c], value U[i,j,c]
        #            for each (j,k) in V_sup[c], value V[j,k,c]
        #            for each (k,i) in W_sup[c], value W[k,i,c]
        
        # To make it linear, fix one variable per term to 1.
        # For term c, fix U[i0,j0,c] = 1 where (i0,j0) = U_sup[c][0]
        # Then unknowns are the remaining U, all V, all W.
        
        # Equations: for each (i,j,k), sum_c [U[i,j,c]*V[j,k,c]*W[k,i,c]] = 1
        # This is still nonlinear because of products.
        
        # Alternative: use a different factorization.
        # Let A_c = U[i,j,c], B_c = V[j,k,c], C_c = W[k,i,c]
        # The equation is sum_c A_c * B_c * C_c = 1
        
        # This is a polynomial system. Hard to solve directly.
        
        # New approach: assume a specific structure where the problem becomes linear.
        # If we fix V[j,k,c] and W[k,i,c] to constants, then it's linear in U.
        
        # Let's try: set all V and W entries to 1 where non-zero, solve for U.
        # Then check if it works.
        
        V_mat = np.zeros((nV, target_rank))
        W_mat = np.zeros((nW, target_rank))
        for c in range(target_rank):
            for (j,k) in V_sup[c]:
                V_mat[j*p+k, c] = 1.0
            for (k,i) in W_sup[c]:
                W_mat[k*n+i, c] = 1.0
        
        # Now solve for U: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in U!
        
        # Build the linear system for U
        # Variables: U[i*m+j, c] for all i,j,c
        # Equations: for each (i,j,k), one equation
        
        num_eq = n * m * p  # 18
        num_vars = nU * target_rank  # 6*17=102 or 6*16=96
        
        # Coefficient matrix A and rhs b
        A = np.zeros((num_eq, num_vars))
        b = np.ones(num_eq)
        
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            for c in range(target_rank):
                coeff = V_mat[j*p+k, c] * W_mat[k*n+i, c]
                var_idx = (i*m + j) * target_rank + c
                A[idx, var_idx] = coeff
        
        # Solve least squares
        try:
            U_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            U_mat = U_sol.reshape(nU, target_rank)
        except:
            continue
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Check if entries are "nice" (integers or half-integers)
            # Scale if needed
            U_list = U_mat.tolist()
            V_list = V_mat.tolist()
            W_list = W_mat.tolist()
            
            # Try to make entries rational
            def clean_matrix(mat):
                cleaned = []
                for row in mat:
                    new_row = []
                    for val in row:
                        if abs(val) < 1e-10:
                            new_row.append(0)
                        else:
                            # Try to round to nearest 0.5
                            rounded = round(val * 2) / 2
                            if abs(val - rounded) < 1e-6:
                                new_row.append(rounded)
                            else:
                                new_row.append(val)
                    cleaned.append(new_row)
                return cleaned
            
            U_list = clean_matrix(U_list)
            V_list = clean_matrix(V_list)
            W_list = clean_matrix(W_list)
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END

