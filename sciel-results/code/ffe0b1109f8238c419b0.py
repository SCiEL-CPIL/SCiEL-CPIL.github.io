import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Candidate ranks
    # We want minimal rank.
    # Try 40, 50, 60, 70, 80.
    # If we find a valid one, return it.
    
    candidate_ranks = [40, 50, 60, 70, 80, 90]
    
    for R in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(R * 100 + attempt)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # ALS Loop
            converged = False
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # For each i', j', solve M u = Y
                # M size: (n*p*p) x R
                # Y size: n*p*p
                for i_prime in range(n):
                    for j_prime in range(m):
                        # Construct M
                        # M[row, r] = W[k, i, r] * V[j_prime, k_prime, r]
                        # row index corresponds to (i, k, k_prime)
                        # i in 0..n-1, k in 0..p-1, k_prime in 0..p-1
                        M = np.zeros((n*p*p, R))
                        Y = np.zeros(n*p*p)
                        
                        # Fill Y
                        # Y[i, k, k_prime] = 1 if i==i_prime and k==k_prime
                        # Note: k is the output col index, k_prime is the B row index?
                        # In V[j', k'], j' is row of B, k' is col of B.
                        # In the equation: sum_r W[k,i,r] U[i',j',r] V[j',k',r]
                        # The RHS is delta(i, i') delta(k, k')?
                        # Wait, the product is A[i',j'] B[j',k'].
                        # This contributes to C[i,k] if i'=i and k'=k.
                        # So RHS is 1 if i==i' and k_prime==k.
                        # My previous Y was delta(k, k_prime). Here k is C col, k_prime is B col.
                        # So yes, delta(k, k_prime).
                        
                        for i in range(n):
                            for k in range(p):
                                for k_prime in range(p):
                                    idx = i * p * p + k * p + k_prime
                                    if i == i_prime and k == k_prime:
                                        Y[idx] = 1.0
                                    
                                    # Fill M
                                    # M[idx, r] = W[k, i, r] * V[j_prime, k_prime, r]
                                    # W index: k*n + i
                                    # V index: j_prime * p + k_prime
                                    W_vec = W[k * n + i, :]
                                    V_vec = V[j_prime * p + k_prime, :]
                                    M[idx, :] = W_vec * V_vec
                        
                        # Solve M u = Y
                        # u is U[i_prime * m + j_prime, :]
                        try:
                            U[i_prime * m + j_prime, :], residuals, rank, sv = np.linalg.lstsq(M, Y, rcond=None)
                        except:
                            pass
                
                # Update V
                # For each j', k', solve
                # Equation: sum_r W[k,i,r] U[i',j',r] V[j',k',r] = delta(i,i') delta(k,k')
                # Fix j', k'. Unknown v_r = V[j',k',r].
                # Term: sum_r v_r W[k,i,r] U[i',j',r]
                # Indices: i, i', k.
                # M size: (n*n*p) x R?
                # i in 0..n-1, i' in 0..n-1, k in 0..p-1.
                # M[idx, r] = W[k, i, r] * U[i_prime, j_prime, r]
                # Y[idx] = 1 if i==i_prime and k==k_prime?
                # Wait, for V update, we fix j', k'.
                # The equation must hold for all i, i', k.
                # RHS is delta(i, i') delta(k, k').
                
                for j_prime in range(m):
                    for k_prime in range(p):
                        M = np.zeros((n*n*p, R))
                        Y = np.zeros(n*n*p)
                        
                        for i in range(n):
                            for i_prime in range(n):
                                for k in range(p):
                                    idx = i * n * p + i_prime * p + k
                                    if i == i_prime and k == k_prime:
                                        Y[idx] = 1.0
                                    
                                    W_vec = W[k * n + i, :]
                                    U_vec = U[i_prime * m + j_prime, :]
                                    M[idx, :] = W_vec * U_vec
                        
                        try:
                            V[j_prime * p + k_prime, :], residuals, rank, sv = np.linalg.lstsq(M, Y, rcond=None)
                        except:
                            pass

                # Update W
                # For each k, i. Unknown w_r = W[k,i,r].
                # Term: sum_r w_r U[i',j',r] V[j',k',r]
                # Indices: i', j', k'.
                # M size: (n*m*p) x R?
                # i' in 0..n-1, j' in 0..m-1, k' in 0..p-1.
                # M[idx, r] = U[i_prime, j_prime, r] * V[j_prime, k_prime, r]
                # Y[idx] = 1 if i_prime==i and k_prime==k?
                # Wait, W is W[k,i].
                # The equation is for fixed k, i.
                # LHS: sum_r w_r U[i',j',r] V[j',k',r].
                # RHS: delta(i', i) delta(k', k).
                
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((n*m*p, R))
                        Y = np.zeros(n*m*p)
                        
                        for i_prime in range(n):
                            for j_prime in range(m):
                                for k_prime in range(p):
                                    idx = i_prime * m * p + j_prime * p + k_prime
                                    if i_prime == i and k_prime == k:
                                        Y[idx] = 1.0
                                    
                                    U_vec = U[i_prime * m + j_prime, :]
                                    V_vec = V[j_prime * p + k_prime, :]
                                    M[idx, :] = U_vec * V_vec
                        
                        try:
                            W[k * n + i, :], residuals, rank, sv = np.linalg.lstsq(M, Y, rcond=None)
                        except:
                            pass
            
            # Check validity
            # Round to nearest integer
            U_int = np.round(U).astype(int)
            V_int = np.round(V).astype(int)
            W_int = np.round(W).astype(int)
            
            # Verify
            valid = True
            for i_prime in range(n):
                for j_prime in range(m):
                    for k_prime in range(p):
                        # Compute sum_r W[k,i,r] U[i',j',r] V[j',k',r]
                        # And check against delta
                        for i in range(n):
                            for k in range(p):
                                val = 0
                                for r in range(R):
                                    val += W_int[k * n + i, r] * U_int[i_prime * m + j_prime, r] * V_int[j_prime * p + k_prime, r]
                                expected = 1 if (i_prime == i and k_prime == k) else 0
                                if val != expected:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
            
            if valid:
                # Return in required format
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
    # Fallback
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
