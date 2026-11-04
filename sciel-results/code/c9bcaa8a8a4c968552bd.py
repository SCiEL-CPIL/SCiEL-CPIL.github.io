import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 130.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # Indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]
    
    # P1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = A00 * (B10 - B00)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = A11 * (B01 - B11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # P6 = (A01 - A11) * (B00 + B01)
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A00 - A10) * (B10 + B11)
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Construct rank 49 decomposition for 4x4x4."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0: continue
                            global_i = bi*2 + li
                            global_j = bj*2 + lj
                            U[global_i*4 + global_j][col] += coeff_a * local_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0: continue
                            global_j = bj*2 + lj
                            global_k = bk*2 + lk
                            V[global_j*4 + global_k][col] += coeff_b * local_v
            
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li][s]
                            if local_w == 0: continue
                            global_k = ck*2 + lk
                            global_i = ci*2 + li
                            W[global_k*4 + global_i][col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=50):
    """Verify the decomposition."""
    import random
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True

def _try_find_rank_49_extension(U4, V4, W4, timeout=30.0):
    """
    Try to find V4_ext (4x49) and W4_ext (4x49) such that
    sum_r U4[i,j,r] * V4_ext[j,r] * W4_ext[i,r] = 1 for all i,j.
    """
    start_time = time.time()
    n = 4
    rank = 49
    
    # Convert to numpy for speed
    U_np = np.array(U4, dtype=float) # 16x49
    
    # We want to solve for V_ext (4x49) and W_ext (4x49)
    # Equation: sum_r U[i*m+j, r] * V_ext[j, r] * W_ext[i, r] = 1
    
    # This is equivalent to:
    # Let M_j be the row j of V_ext (size 49).
    # Let N_i be the row i of W_ext (size 49).
    # sum_r U[i*4+j, r] M_j[r] N_i[r] = 1
    
    # This is a system of 16 equations.
    # Variables: 49*4 + 49*4 = 392 variables.
    # It is bilinear. We can use random restarts with coordinate descent or simple gradient descent.
    # Or simply: fix V_ext, solve for W_ext linearly?
    # If V_ext is fixed, the equation is linear in W_ext?
    # sum_r (U[i*4+j, r] * V_ext[j, r]) * W_ext[i, r] = 1.
    # This is NOT linear in W_ext because W_ext[i, r] is multiplied by a term that depends on r.
    # Wait. For a fixed i, the equation involves W_ext[i, r] for all r.
    # Let A_ij[r] = U[i*4+j, r] * V_ext[j, r].
    # Then sum_r A_ij[r] * W_ext[i, r] = 1.
    # This is linear in the vector W_ext[i, :] for each i, but coupled across i?
    # No, for each i, we have 4 equations (one for each j).
    # sum_r A_ij[r] * W_ext[i, r] = 1 for j=0..3.
    # This is a system of 4 linear equations in 49 unknowns (W_ext[i, 0..48]).
    # So if we fix V_ext, we can solve for W_ext[i, :] for each i independently!
    
    # So the strategy is:
    # 1. Randomly initialize V_ext (4x49).
    # 2. For each i, solve the linear system for W_ext[i, :].
    #    The system is: A_i * W_ext[i, :]^T = 1 (vector of ones).
    #    Where A_i is 4x49, with A_i[j, r] = U[i*4+j, r] * V_ext[j, r].
    # 3. Check if the resulting W_ext satisfies the conditions for all i,j.
    #    Actually, if we solve for W_ext[i] to satisfy the equations for that i,
    #    does it automatically satisfy the tensor condition?
    #    The condition is exactly sum_r U V W = 1.
    #    So yes, if we solve the linear system for each i, we get a valid W_ext.
    #    However, we must ensure that the solution exists (rank of A_i is sufficient).
    #    Since we have 49 unknowns and 4 equations, a solution always exists unless the equations are contradictory (e.g. 0=1).
    #    With random V_ext, A_i will likely have full rank 4.
    
    # But wait, we need V_ext and W_ext to be part of the SAME decomposition.
    # The decomposition is valid if the tensor equation holds.
    # If we construct W_ext based on V_ext, the equation holds by construction.
    # So ANY non-degenerate V_ext will yield a valid W_ext?
    # Let's verify.
    # We need sum_r U[i,j,r] V[j,r] W[i,r] = 1.
    # If we pick V arbitrarily, and set W[i,r] to be the solution of the linear system:
    # sum_r (U[i,j,r] V[j,r]) W[i,r] = 1 for all j.
    # Then the condition is satisfied.
    # So Rank 49 is ALWAYS possible?
    # Why did I think it might not be?
    # Ah, the entries must be "integer/half-integer".
    # The problem statement says: "factor matrices ... with integer/half-integer entries".
    # If we use random floats, the entries are not integers/half-integers.
    # We need to find V_ext and W_ext with entries in {0, 0.5, 1, 1.5, ...} or similar?
    # "integer/half-integer" usually means $k/2$ for integer $k$.
    
    # So we need to search for V_ext and W_ext in a discrete grid.
    # This makes it a discrete search problem.
    
    # Given the time limit, random search over a grid might be too slow if the grid is large.
    # However, we can try a few random integer/half-integer vectors for V_ext.
    
    for attempt in range(500):
        if time.time() - start_time > timeout:
            break
            
        # Random V_ext with values in {-1, -0.5, 0, 0.5, 1}
        V_ext = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(4, 49))
        
        # Solve for W_ext
        W_ext = np.zeros((4, 49))
        valid = True
        for i in range(4):
            # Build matrix A_i (4x49)
            A_i = np.zeros((4, 49))
            for j in range(4):
                for r in range(49):
                    A_i[j, r] = U_np[i*4+j, r] * V_ext[j, r]
            
            # Solve A_i * W_ext[i, :]^T = ones(4)
            # Using least squares
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A_i, np.ones(4), rcond=None)
                # Check if solution is exact
                if np.allclose(A_i @ sol, np.ones(4), atol=1e-9):
                    W_ext[i, :] = sol
                else:
                    valid = False
                    break
            except:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check if W_ext entries are close to half-integers
        # Round them and check if the rounded version still works
        W_ext_rounded = np.round(W_ext * 2) / 2
        
        # Verify with rounded values
        ok = True
        for i in range(4):
            for j in range(4):
                val = 0
                for r in range(49):
                    val += U_np[i*4+j, r] * V_ext[j, r] * W_ext_rounded[i, r]
                if abs(val - 1) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            # Also check if V_ext entries are half-integers (they are by construction)
            # Construct the full decomposition
            # U is U4
            # V is V4 extended with V_ext at k=4
            # W is W4 extended with W_ext at k=4
            
            # V shape: 20x49. V[j*5+k, r].
            # For k=4, index is 5j+4.
            # So V[5j+4, r] = V_ext[j, r].
            
            # W shape: 20x49. W[k*4+i, r].
            # For k=4, index is 4*4+i = 16+i.
            # So W[16+i, r] = W_ext[i, r].
            
            return V_ext, W_ext_rounded
            
    return None

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    try:
        result_444 = _strassen_4x4x4()
        
        U4 = result_444["U"]
        V4 = result_444["V"]
        W4 = result_444["W"]
        rank_444 = len(U4[0])
        
        # Try to find rank 49 extension
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time > 10:
            ext = _try_find_rank_49_extension(U4, V4, W4, timeout=min(remaining_time - 10, 20))
            if ext is not None:
                V_ext, W_ext = ext
                new_rank = rank_444
                U = [row[:] for row in U4]
                V = [[0] * new_rank for _ in range(m * p)]
                W = [[0] * new_rank for _ in range(p * n)]
                
                # Copy V4
                for j in range(4):
                    for k in range(4):
                        for r in range(rank_444):
                            V[j*5 + k][r] = V4[j*4 + k][r]
                
                # Copy W4
                for k in range(4):
                    for i in range(4):
                        for r in range(rank_444):
                            W[k*4 + i][r] = W4[k*4 + i][r]
                            
                # Add extension
                for j in range(4):
                    for r in range(rank_444):
                        V[j*5 + 4][r] = V_ext[j][r]
                        
                for i in range(4):
                    for r in range(rank_444):
                        W[4*4 + i][r] = W_ext[i][r]
                        
                result = {"U": U, "V": V, "W": W}
                
                if _verify_decomposition(result, n, m, p):
                    return result
        
        # Fallback to Rank 50
        new_rank = rank_444 + 1
        U = [[0] * new_rank for _ in range(n * m)]
        V = [[0] * new_rank for _ in range(m * p)]
        W = [[0] * new_rank for _ in range(p * n)]
        
        for i in range(16):
            for r in range(rank_444):
                U[i][r] = U4[i][r]
                
        for j in range(4):
            for k in range(4):
                for r in range(rank_444):
                    V[j*5 + k][r] = V4[j*4 + k][r]
                    
        for k in range(4):
            for i in range(4):
                for r in range(rank_444):
                    W[k*4 + i][r] = W4[k*4 + i][r]
                    
        new_col = rank_444
        for i in range(4):
            for j in range(4):
                U[i*4 + j][new_col] = 1
            V[j*5 + 4][new_col] = 1
            W[4*4 + i][new_col] = 1
        
        result = {"U": U, "V": V, "W": W}
        
        if _verify_decomposition(result, n, m, p):
            return result
            
    except Exception as e:
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
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

# EVOLVE-BLOCK-END
