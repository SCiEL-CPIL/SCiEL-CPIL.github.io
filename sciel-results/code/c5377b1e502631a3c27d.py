import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook solution as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    # Search for lower rank
    # We know rank must be at least max(nm, mp, pn) = max(15, 30, 18) = 30?
    # Actually, lower bound is max(nm, mp, pn) is not correct.
    # Lower bound is max(n+m-1, m+p-1, p+n-1) for univariate? No.
    # For matrix mult, lower bound is max(nm, mp, pn) is not tight.
    # Strassen bound: $r \ge \frac{(n+m-1)(m+p-1)(p+n-1)}{nmp} \times nmp$? No.
    # Let's just search from 89 down to 70.
    
    for r in range(rank_schoolbook - 1, 60, -1):
        if time.time() > deadline:
            break
            
        # Try a few random configurations for this rank
        max_tries = 50
        for _ in range(max_tries):
            if time.time() > deadline:
                break
                
            # Random U and V with small integers
            # Using {-1, 0, 1} or {-2, -1, 0, 1, 2}
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            
            # Construct linear system for W
            # Variables: W_{k, i, c} for k in [0,p), i in [0,n), c in [0,r)
            # Total variables: p * n * r
            # Equations: for each i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # We can reshape W to a vector of length p*n*r
            # Index for W variable: (k * n + i) * r + c
            # Equation index: (i * m + j) * p + k
            
            num_eqs = n * m * p
            num_vars = p * n * r
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Fill A
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        # Coeff for W[k, i, c] is U[i, j, c] * V[j, k, c]
                        # W index in vector: (k * n + i) * r + c
                        for c in range(r):
                            var_idx = (k * n + i) * r + c
                            A[eq_idx, var_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            # Solve A * W_vec = b
            # Use least squares
            W_vec, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check residual
            if residuals is not None and residuals[0] < 1e-6:
                # Reshape W_vec to matrix
                W = W_vec.reshape((p * n, r))
                
                # Round to nearest half-integer
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W_rounded[k * n + i, c]
                            if abs(val - 1) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Convert to lists
                    U_list = U.astype(int).tolist()
                    V_list = V.astype(int).tolist()
                    # W_rounded might have .5
                    W_list = W_rounded.tolist()
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
            
    return best_result

