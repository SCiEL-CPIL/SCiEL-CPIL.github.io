import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Try ranks from lower bound up to 23
    # Lower bound: max(nm, mp, pn) is a weak bound, but rank >= max(nm, mp, pn) is not true.
    # Rank >= nm is true? No.
    # Let's just try 12 to 23.
    
    best_rank = 24
    best_solution = None
    
    # Fallback: Schoolbook
    U_school = np.zeros((nm, 24), dtype=int)
    V_school = np.zeros((mp, 24), dtype=int)
    W_school = np.zeros((pn, 24), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    for r in range(12, 24):
        if time.time() > deadline - 10:
            break
            
        # Number of trials per rank
        trials = 5000
        if r < 15:
            trials = 20000
        elif r < 18:
            trials = 10000
            
        for _ in range(trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V with sparse entries
            # Use 0, 1, -1, 0.5, -0.5
            # Sparsity ~ 30-50%
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            for i in range(nm):
                for j in range(r):
                    if random.random() < 0.4:
                        U[i, j] = random.choice([1, -1, 0.5, -0.5])
                        
            for i in range(mp):
                for j in range(r):
                    if random.random() < 0.4:
                        V[i, j] = random.choice([1, -1, 0.5, -0.5])
            
            # Check if we can find W
            # For each c, solve A_c w_c = z_c
            # A_c is (nm*mp) x r
            # z_c is (nm*mp)
            
            W_found = True
            W_sol = np.zeros((pn, r))
            
            for c in range(pn):
                # Construct A_c
                # A_c[idx, alpha] = U[a, alpha] * V[b, alpha]
                # idx corresponds to (a, b)
                
                # Flatten U and V to create the matrix
                # We can use np.einsum or manual loop
                # U is (nm, r), V is (mp, r)
                # We want a matrix of size (nm*mp, r)
                # Entry (a, b, alpha) = U[a, alpha] * V[b, alpha]
                
                # Efficiently:
                # A_c = np.zeros((nm*mp, r))
                # for a in range(nm):
                #   for b in range(mp):
                #     for alpha in range(r):
                #       A_c[a*mp+b, alpha] = U[a, alpha] * V[b, alpha]
                
                # Vectorized:
                # U_exp = U[:, np.newaxis, :] # (nm, 1, r)
                # V_exp = V[np.newaxis, :, :] # (1, mp, r)
                # Prod = U_exp * V_exp # (nm, mp, r)
                # A_c = Prod.reshape(nm*mp, r)
                
                A_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, r)
                z_c = T[:, :, c].flatten()
                
                # Solve A_c w = z_c
                # Check if z_c is in column space of A_c
                # Use least squares and check residual
                try:
                    w_c, residuals, rank, sv = np.linalg.lstsq(A_c, z_c, rcond=None)
                    # Check residual
                    if residuals.size > 0:
                        if residuals[0] > 1e-9:
                            W_found = False
                            break
                    else:
                        # Underdetermined, check if A_c w_c == z_c
                        if not np.allclose(A_c @ w_c, z_c, atol=1e-9):
                            W_found = False
                            break
                    
                    # Check if w_c has half-integer entries
                    # Multiply by 2 and check if close to integer
                    w_c_scaled = w_c * 2
                    if not np.allclose(w_c_scaled, np.round(w_c_scaled), atol=1e-6):
                        W_found = False
                        break
                        
                    W_sol[c, :] = w_c
                    
                except:
                    W_found = False
                    break
            
            if W_found:
                # Verify solution
                # Reconstruct tensor
                # Sum_alpha U[a, alpha] V[b, alpha] W[c, alpha]
                # U (nm, r), V (mp, r), W (pn, r)
                # Result (nm, mp, pn)
                
                # Use einsum
                # 'ar, br, cr -> abc'
                reconstructed = np.einsum('ar,br,cr->abc', U, V, W_sol)
                
                if np.allclose(reconstructed, T, atol=1e-6):
                    # Convert to lists of fractions or floats
                    # The prompt accepts integer/half-integer
                    # We return floats for simplicity, or fractions?
                    # "integer/half-integer entries"
                    # We can return floats that are exactly 0.5, 1.0 etc.
                    
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W_sol]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    U_list = [[int(x) for x in row] for row in U_school]
    V_list = [[int(x) for x in row] for row in V_school]
    W_list = [[int(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
