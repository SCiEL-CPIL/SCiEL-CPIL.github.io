import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    DEADLINE = 135

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook baseline
    rank_school = n * m * p
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_U = U
    best_V = V
    best_W = W
    best_rank = rank_school

    def verify(U, V, W, rank):
        # U, V, W are numpy arrays
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Use float for speed, then check closeness
        # We need EXACT integer/half-integer verification.
        # Since entries are half-integers, 2*entries are integers.
        # We can check if 8 * sum == 8? No.
        # Let's just use float sum and check if it's very close to 1.
        # Then do a final exact check if needed.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_exact(U, V, W, rank):
        # Convert to integers by multiplying by 2
        U_int = (U * 2).round().astype(int)
        V_int = (V * 2).round().astype(int)
        W_int = (W * 2).round().astype(int)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                    if s != 4: # 1.0 * 2 * 2 * 2? No.
                         # U, V, W are half-integers.
                         # Product is quarter-integer? No.
                         # (1/2) * (1/2) * (1/2) = 1/8.
                         # Sum of 1/8s.
                         # We want sum = 1 = 8/8.
                         # So sum of (U_int * V_int * W_int) should be 8.
                         pass
        
        # Let's re-evaluate the scaling.
        # U = u/2, V = v/2, W = w/2.
        # Sum (u/2)(v/2)(w/2) = 1/8 Sum(uvw) = 1 => Sum(uvw) = 8.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                    if s != 8:
                        return False
        return True

    # Search for lower ranks
    # Try ranks from 30 to 53
    for target_rank in range(30, best_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Random half-integers for U and V
            # Range [-2, 2] usually sufficient for small ranks
            U_rand = np.random.randint(-4, 5, size=(nm, target_rank)) / 2.0
            V_rand = np.random.randint(-4, 5, size=(mp, target_rank)) / 2.0
            
            # Build linear system for W
            # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            # Total vars: p*n*R
            num_vars = pn * target_rank
            num_eqs = nm * p # n*m*p
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Map indices
            # Eq idx: i*m + j, then k? 
            # Let's flatten eqs as (i, j, k) -> i*m*p + j*p + k
            # Var idx: (k, i, r) -> k*n*R + i*R + r
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(target_rank):
                            var_idx = k * n * target_rank + i * target_rank + r
                            A[eq_idx, var_idx] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
            
            # Solve least squares
            W_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Reshape W_sol
            W_sol = W_sol.reshape(p, n, target_rank)
            W_sol = W_sol.transpose(1, 0, 2) # (n, p, R)
            W_sol = W_sol.reshape(pn, target_rank) # (p*n, R)
            
            # Round W_sol to half-integers
            W_rounded = np.round(W_sol * 2) / 2.0
            
            # Check error
            # We can quickly check the max error
            # sum_r U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rounded[k*n+i, r]
            
            # If error is small, verify exact
            # To save time, just verify exact directly if rank is small enough
            if verify_exact(U_rand, V_rand, W_rounded, target_rank):
                best_U = U_rand
                best_V = V_rand
                best_W = W_rounded
                best_rank = target_rank
                found = True
                break
                
        if found:
            # Found a solution for this rank, try lower ranks
            continue
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
