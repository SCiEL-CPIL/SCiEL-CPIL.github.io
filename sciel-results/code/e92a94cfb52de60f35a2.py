"""
Strategy: Deterministic construction of a low-rank decomposition for 2x4x4 matrix multiplication.
1. Base case: Use Strassen's algorithm (rank 7) for 2x2x2 blocks.
   Tiling the 2x4x4 problem into four 2x2x2 blocks (since 4=2+2) yields a rank of 4 * 7 = 28.
   This is a valid, known decomposition.
2. Optimization: Attempt to find a decomposition of rank < 28 by exploiting the 
   rectangular structure. Specifically, we look for a decomposition where the 
   "cross terms" between the block rows/cols can be shared or simplified.
   
   However, finding a rank < 28 exactly with integer/half-integer coefficients is a 
   hard algebraic problem. The current best known bounds for 2x4x4 are around 25-26.
   
   To ensure a valid solution (score > 0) and potentially beat the schoolbook (32), 
   we will primarily rely on the robust Strassen tiling (rank 28). 
   
   We will also attempt a quick randomized search for rank 25-27 using the 
   "fix U, V, solve W" linear system approach, but with a much tighter time budget 
   and smarter initialization (using the Strassen structure as a starting point or 
   perturbation). If found, we use it. Otherwise, we return the Strassen rank 28.
"""

import time
import numpy as np
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # 150s limit, keep 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # --- 1. Construct the Strassen Rank-28 Decomposition ---
    # This is a guaranteed valid solution with rank 28.
    # We construct it by tiling the 2x4 and 4x4 matrices into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2? 
    # Wait, A is 2x4. So A = [A_left, A_right] where A_left, A_right are 2x2.
    # B is 4x4. B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A * B = [A_left, A_right] * [[B11, B12], [B21, B22]]
    # C = [ A_left*B11 + A_right*B21,  A_left*B12 + A_right*B22 ]
    # C is 2x4. Let C = [C_left, C_right].
    
    # We need to compute 4 matrix products of 2x2 matrices:
    # P1 = A_left * B11
    # P2 = A_right * B21
    # P3 = A_left * B12
    # P4 = A_right * B22
    
    # Each 2x2 product can be done with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Strassen factors for 2x2x2:
    # U, V, W are 4x7 matrices (since 2x2=4 entries).
    # Indices for U: (i, j) -> i*2+j. i,j in {0,1}.
    # Indices for V: (j, k) -> j*2+k. j,k in {0,1}.
    # Indices for W: (k, i) -> k*2+i. k,i in {0,1}.
    
    # Standard Strassen decomposition (one common set of integer coefficients):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the U, V, W factors for a single 2x2x2 block.
    # U[i,j,r], V[j,k,r], W[k,i,r]
    
    # r=0: M1
    U_s = np.zeros((2, 2, 7), dtype=float)
    V_s = np.zeros((2, 2, 7), dtype=float)
    W_s = np.zeros((2, 2, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    # U[0,0,0]=1, U[1,1,0]=1
    # V[0,0,0]=1, V[1,1,0]=1
    # W[0,0,0]=1 (C11), W[1,1,0]=1 (C22) -- wait, C22 = M1 - M2 + M3 + M6
    # Let's map carefully.
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # r=0 (M1):
    U_s[0,0,0] = 1; U_s[1,1,0] = 1
    V_s[0,0,0] = 1; V_s[1,1,0] = 1
    W_s[0,0,0] = 1 # C11
    W_s[1,1,0] = 1 # C22
    
    # r=1 (M2 = (A21+A22)B11):
    U_s[1,0,1] = 1; U_s[1,1,1] = 1
    V_s[0,0,1] = 1
    W_s[0,1,1] = 1 # C21
    
    # r=2 (M3 = A11(B12-B22)):
    U_s[0,0,2] = 1
    V_s[0,1,2] = 1; V_s[1,1,2] = -1
    W_s[0,1,2] = 1 # C12
    W_s[1,1,2] = 1 # C22
    
    # r=3 (M4 = A22(B21-B11)):
    U_s[1,1,3] = 1
    V_s[1,0,3] = 1; V_s[0,0,3] = -1
    W_s[0,0,3] = 1 # C11
    W_s[0,1,3] = 1 # C21
    
    # r=4 (M5 = (A11+A12)B22):
    U_s[0,0,4] = 1; U_s[0,1,4] = 1
    V_s[1,1,4] = 1
    W_s[0,0,4] = -1 # C11
    W_s[0,1,4] = 1 # C12
    
    # r=5 (M6 = (A21-A11)(B11+B12)):
    U_s[1,0,5] = 1; U_s[0,0,5] = -1
    V_s[0,0,5] = 1; V_s[0,1,5] = 1
    W_s[1,1,5] = 1 # C22
    
    # r=6 (M7 = (A12-A22)(B21+B22)):
    U_s[0,1,6] = 1; U_s[1,1,6] = -1
    V_s[1,0,6] = 1; V_s[1,1,6] = 1
    W_s[0,0,6] = 1 # C11
    
    # Verify Strassen block
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Should be 1 for all i,j,k in {0,1}
    T_block = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U_s[i,j,r] * V_s[j,k,r] * W_s[k,i,r]
                T_block[i,j,k] = s
    if not np.allclose(T_block, 1.0):
        # Fallback to schoolbook for block if Strassen construction has error
        # (Should not happen, but safe)
        U_s = np.zeros((2,2,4)); V_s = np.zeros((2,2,4)); W_s = np.zeros((2,2,4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i*4 + j*2 + k
                    U_s[i,j,r] = 1
                    V_s[j,k,r] = 1
                    W_s[k,i,r] = 1

    # Now assemble the 2x4x4 decomposition using 4 blocks.
    # A is 2x4. A = [A_L, A_R] (2x2 each)
    # B is 4x4. B = [[B_11, B_12], [B_21, B_22]] (2x2 each)
    # C is 2x4. C = [C_L, C_R] (2x2 each)
    
    # C_L = A_L * B_11 + A_R * B_21
    # C_R = A_L * B_12 + A_R * B_22
    
    # We have 4 products to compute:
    # P1 = A_L * B_11 (uses A_L, B_11, contributes to C_L)
    # P2 = A_R * B_21 (uses A_R, B_21, contributes to C_L)
    # P3 = A_L * B_12 (uses A_L, B_12, contributes to C_R)
    # P4 = A_R * B_22 (uses A_R, B_22, contributes to C_R)
    
    # Total Rank R = 4 * 7 = 28.
    R_total = 28
    
    U_full = np.zeros((n*m, R_total))
    V_full = np.zeros((m*p, R_total))
    W_full = np.zeros((p*n, R_total))
    
    # Mapping of indices:
    # A indices: i in {0,1}, j in {0,3}. 
    #   j=0,1 -> A_L. j=2,3 -> A_R.
    # B indices: j in {0,3}, k in {0,3}.
    #   j=0,1, k=0,1 -> B_11
    #   j=0,1, k=2,3 -> B_12
    #   j=2,3, k=0,1 -> B_21
    #   j=2,3, k=2,3 -> B_22
    # C indices: k in {0,3}, i in {0,1}.
    #   k=0,1 -> C_L. k=2,3 -> C_R.
    
    # Block 1: A_L * B_11 -> C_L
    # A_L is A[0:2, 0:2]. B_11 is B[0:2, 0:2]. C_L is C[0:2, 0:2].
    # Strassen U_s[i,j,r] corresponds to A_L[i,j].
    # Strassen V_s[j,k,r] corresponds to B_11[j,k].
    # Strassen W_s[k,i,r] corresponds to C_L[k,i].
    
    for r in range(7):
        r_global = r # 0-6
        
        # Fill U
        for i in range(2):
            for j in range(2):
                U_full[i*m + j, r_global] = U_s[i,j,r]
        
        # Fill V
        for j in range(2):
            for k in range(2):
                V_full[j*p + k, r_global] = V_s[j,k,r]
        
        # Fill W
        for k in range(2):
            for i in range(2):
                W_full[k*n + i, r_global] = W_s[k,i,r]
    
    # Block 2: A_R * B_21 -> C_L
    # A_R is A[0:2, 2:4]. B_21 is B[2:4, 0:2]. C_L is C[0:2, 0:2].
    # Note: C_L is the same output location as Block 1.
    # A_R indices: i in {0,1}, j in {2,3}. Let j' = j-2.
    # B_21 indices: j in {2,3}, k in {0,1}. Let j' = j-2.
    # C_L indices: k in {0,1}, i in {0,1}.
    
    for r in range(7):
        r_global = 7 + r # 7-13
        
        # Fill U (A_R)
        for i in range(2):
            for j_prime in range(2):
                j = 2 + j_prime
                U_full[i*m + j, r_global] = U_s[i,j_prime,r]
        
        # Fill V (B_21)
        for j_prime in range(2):
            for k in range(2):
                j = 2 + j_prime
                V_full[j*p + k, r_global] = V_s[j_prime,k,r]
        
        # Fill W (C_L)
        for k in range(2):
            for i in range(2):
                W_full[k*n + i, r_global] = W_s[k,i,r]
    
    # Block 3: A_L * B_12 -> C_R
    # A_L is A[0:2, 0:2]. B_12 is B[0:2, 2:4]. C_R is C[2:4, 0:2].
    # C_R indices: k in {2,3}, i in {0,1}. Let k' = k-2.
    
    for r in range(7):
        r_global = 14 + r # 14-20
        
        # Fill U (A_L)
        for i in range(2):
            for j in range(2):
                U_full[i*m + j, r_global] = U_s[i,j,r]
        
        # Fill V (B_12)
        for j in range(2):
            for k_prime in range(2):
                k = 2 + k_prime
                V_full[j*p + k, r_global] = V_s[j,k_prime,r]
        
        # Fill W (C_R)
        for k_prime in range(2):
            for i in range(2):
                k = 2 + k_prime
                W_full[k*n + i, r_global] = W_s[k_prime,i,r]
    
    # Block 4: A_R * B_22 -> C_R
    # A_R is A[0:2, 2:4]. B_22 is B[2:4, 2:4]. C_R is C[2:4, 0:2].
    
    for r in range(7):
        r_global = 21 + r # 21-27
        
        # Fill U (A_R)
        for i in range(2):
            for j_prime in range(2):
                j = 2 + j_prime
                U_full[i*m + j, r_global] = U_s[i,j_prime,r]
        
        # Fill V (B_22)
        for j_prime in range(2):
            for k_prime in range(2):
                j = 2 + j_prime
                k = 2 + k_prime
                V_full[j*p + k, r_global] = V_s[j_prime,k_prime,r]
        
        # Fill W (C_R)
        for k_prime in range(2):
            for i in range(2):
                k = 2 + k_prime
                W_full[k*n + i, r_global] = W_s[k_prime,i,r]
    
    # Verify the constructed decomposition
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R_total):
                    s += U_full[i*m + j, r] * V_full[j*p + k, r] * W_full[k*n + i, r]
                T_recon[i, j, k] = s
    
    if not np.allclose(T_recon, 1.0, atol=1e-9):
        # If Strassen assembly failed (shouldn't), fall back to schoolbook
        R_school = n * m * p
        U_school = np.zeros((n*m, R_school))
        V_school = np.zeros((m*p, R_school))
        W_school = np.zeros((p*n, R_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U = U_full
    best_V = V_full
    best_W = W_full
    best_rank = R_total
    
    # --- 2. Attempt to find a lower rank (25-27) ---
    # We will use a randomized search for ranks 27, 26, 25.
    # Strategy: Fix U and V randomly (or structured), solve for W.
    # If W is half-integer, we have a solution.
    
    def is_half_integer(x, tol=1e-9):
        return np.allclose(x * 2, np.round(x * 2), atol=tol)

    def try_rank_optimized(R, max_attempts=200):
        # U: (n, m, R)
        # V: (m, p, R)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Use random sparse integers
            U = np.random.randint(-1, 2, size=(n, m, R))
            V = np.random.randint(-1, 2, size=(m, p, R))
            
            valid = True
            W = np.zeros((p, n, R))
            
            for k in range(p):
                for i in range(n):
                    # Build M_{k,i} of size (m, R)
                    # M[j, r] = U[i, j, r] * V[j, k, r]
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i, j, :] * V[j, k, :]
                    
                    # Solve M @ W_ki = 1
                    try:
                        if R == m:
                            if abs(np.linalg.det(M)) < 1e-9:
                                valid = False
                                break
                            W_ki = np.linalg.solve(M, np.ones(m))
                        else:
                            # Least squares
                            W_ki, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            if rank < m:
                                valid = False
                                break
                            if not np.allclose(M @ W_ki, np.ones(m), atol=1e-6):
                                valid = False
                                break
                        
                        if not is_half_integer(W_ki):
                            valid = False
                            break
                            
                        W[k, i, :] = W_ki
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to required format
                U_out = np.zeros((n*m, R))
                V_out = np.zeros((m*p, R))
                W_out = np.zeros((p*n, R))
                
                for i in range(n):
                    for j in range(m):
                        U_out[i*m + j, :] = U[i, j, :]
                
                for j in range(m):
                    for k in range(p):
                        V_out[j*p + k, :] = V[j, k, :]
                
                for k in range(p):
                    for i in range(n):
                        W_out[k*n + i, :] = W[k, i, :]
                
                return U_out, V_out, W_out
        
        return None

    # Try ranks 27, 26, 25
    for R in [27, 26, 25]:
        if time.time() > deadline - 5:
            break
        result = try_rank_optimized(R, max_attempts=50)
        if result:
            U_out, V_out, W_out = result
            # Verify
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_out[i*m + j, r] * V_out[j*p + k, r] * W_out[k*n + i, r]
                        T_recon[i, j, k] = val
            
            if np.allclose(T_recon, 1.0, atol=1e-9):
                best_U = U_out
                best_V = V_out
                best_W = W_out
                best_rank = R
                break # Found lower rank, stop

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
