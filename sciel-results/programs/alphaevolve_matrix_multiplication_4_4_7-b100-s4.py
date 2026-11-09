import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Part 1: 4x4x4 decomposition using Strassen (Rank 49)
    # Part 2: 4x4x3 decomposition using Schoolbook (Rank 48)
    rank_444 = 49
    rank_443 = 44
    total_rank = rank_444 + rank_443
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 1: 4x4x4 ---
    # We decompose the 4x4 multiplication into 7 Strassen products of 2x2 blocks.
    # Each 2x2 product is decomposed into 7 Winograd (Strassen) products of scalars.
    # Total 49 scalar products.
    
    # Strassen definitions for 2x2 block multiplication
    # X is 2x2, Y is 2x2, Z is 2x2
    # s: index 0-6
    # X_def: list of (row, col, coeff) for X
    # Y_def: list of (row, col, coeff) for Y
    # Z_def: list of (row, col, coeff) for Z
    
    strassen_2x2 = [
        # 0: M1 = (X00 + X11)(Y00 + Y11) -> Z00 += M1, Z11 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 1: M2 = (X10 + X11)Y00 -> Z10 += M2, Z11 -= M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # 2: M3 = X00(Y01 - Y11) -> Z01 += M3, Z11 += M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # 3: M4 = X11(Y10 - Y00) -> Z00 += M4, Z10 += M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # 4: M5 = (X00 + X01)Y11 -> Z00 -= M5, Z01 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # 5: M6 = (X10 - X00)(Y00 + Y01) -> Z11 += M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # 6: M7 = (X01 - X11)(Y10 + Y11) -> Z00 += M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Note: The Winograd algorithm for 2x2 scalar multiplication is the same as Strassen.
    # We reuse the same definitions for the inner 2x2 multiplications.
    
    for s_idx, (X_def, Y_def, Z_def) in enumerate(strassen_2x2):
        for w_idx, (u_def, v_def, out_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx
            
            # U column: Coefficients for A
            # A is 4x4. Block (bx, by) is A[2*bx:2*bx+2, 2*by:2*by+2].
            # X is 2x2. X[i', j'] corresponds to block (bx, by) in the Strassen formula.
            # u is a linear form of X. u_val = sum u[i', j'] X[i', j'].
            # So A[2*bx + i', 2*by + j'] has coeff X_coeff * u_coeff.
            
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col
            
            # V column: Coefficients for B
            # B is 4x4. Block (bx, by) is B[2*bx:2*bx+2, 2*by:2*by+2].
            # Y is 2x2. Y[i', j'] corresponds to block (bx, by).
            # v is a linear form of Y. v_val = sum v[i', j'] Y[i', j'].
            # So B[2*bx + i', 2*by + j'] has coeff Y_coeff * v_coeff.
            
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col
            
            # W column: Coefficients for C
            # C is 4x4. Block (dx, dy) is C[2*dx:2*dx+2, 2*dy:2*dy+2].
            # Z is 2x2. Z[i', j'] corresponds to block (dx, dy).
            # out is a linear form of Z. out_val = sum out[i', j'] Z[i', j'].
            # So C[2*dx + i', 2*dy + j'] has coeff Z_coeff * out_coeff.
            # C index: c = col * n + row.
            
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (i_prime, j_prime, co) in out_def:
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col
            
    # --- Part 2: 4x4x3 ---
    # Improved: Use Strassen for first 2 cols of each (2x2)*(2x3) subproblem + schoolbook for 3rd col.
    # A is 4x4, B2 is 4x3 (cols 4-6 of full B), C2 is 4x3 (cols 4-6 of full C).
    # Partition A into four 2x2 blocks: A[r1:r1+2, c1:c1+2] for r1,c1 in {0,2}
    # Partition B2 into two 2x3 blocks: B2[0:2, :] and B2[2:4, :]
    # C2[0:2, :] = A[0:2,0:2]*B2[0:2,:] + A[0:2,2:4]*B2[2:4,:]
    # C2[2:4, :] = A[2:4,0:2]*B2[0:2,:] + A[2:4,2:4]*B2[2:4,:]
    # Each (2x2)*(2x3) uses rank 11 (7 Strassen + 4 schoolbook)
    # Total: 4 * 11 = 44
    
    rank_443 = 44
    total_rank = rank_444 + rank_443
    
    # Reallocate
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Rebuild Part 1 (same as before)
    for s_idx, (X_def, Y_def, Z_def) in enumerate(strassen_2x2):
        for w_idx, (u_def, v_def, out_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx
            
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col
            
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col
            
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (i_prime, j_prime, co) in out_def:
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col
    
    # Part 2: 4x4x3 with rank-44 decomposition
    offset = rank_444
    col = 0
    
    # Strassen 2x2 definitions (for the first 2 columns of the 2x3 product)
    # For (2x2)*(2x2) where the 2x2 B is the first 2 cols of the 2x3 B:
    # B2_sub = [[B00, B01], [B10, B11]] (local indices)
    # C_sub = [[C00, C01], [C10, C11]] (local indices)
    
    # Strassen terms for (2x2)*(2x2):
    # Each term: (U_coeffs for A 2x2, V_coeffs for B 2x2, W_coeffs for C 2x2)
    strassen_terms = [
        # (U: [(row,col,coeff)], V: [(row,col,coeff)], W: [(row,col,coeff)])
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # P1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),           # P2
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),            # P3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),            # P4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),            # P5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),            # P6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),            # P7
    ]
    
    # For each of the 4 (2x2)*(2x3) subproblems:
    # A block is A[ar:ar+2, ac:ac+2] for (ar, ac) in [(0,0),(0,2),(2,0),(2,2)]
    # B block is B2[br:br+2, :] for (br) in [0, 2]
    # C block is C2[cr:cr+2, :] for (cr) in [0, 2]
    # C2[cr:cr+2, :] = A[ar:ar+2, ac:ac+2] * B2[br:br+2, :]
    
    # Mapping:
    # C2 row 0-1 comes from A rows 0-1 * B2 rows 0-1 + A rows 0-1 * B2 rows 2-3
    # C2 row 2-3 comes from A rows 2-3 * B2 rows 0-1 + A rows 2-3 * B2 rows 2-3
    
    # So the 4 subproblems are:
    # (ar, ac, br, cr) = (0,0,0,0), (0,2,2,0), (2,0,0,2), (2,2,2,2)
    
    subproblems = [
        (0, 0, 0, 0),  # A[0:2,0:2] * B2[0:2,:] -> C2[0:2,:]
        (0, 2, 2, 0),  # A[0:2,2:4] * B2[2:4,:] -> C2[0:2,:]
        (2, 0, 0, 2),  # A[2:4,0:2] * B2[0:2,:] -> C2[2:4,:]
        (2, 2, 2, 2),  # A[2:4,2:4] * B2[2:4,:] -> C2[2:4,:]
    ]
    
    for (ar, ac, br, cr) in subproblems:
        # Strassen terms for first 2 columns (local cols 0,1 of B2, which are global cols 4,5)
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            U_col = np.zeros(nm)
            V_col = np.zeros(mp)
            W_col = np.zeros(pn)
            
            # U: A[ar:ar+2, ac:ac+2]
            for (lr, lc, coeff) in u_coeffs:
                i = ar + lr
                j = ac + lc
                a_idx = i * m + j
                U_col[a_idx] += coeff
            
            # V: B2[br:br+2, 0:2] -> global B[br:br+2, 4:6]
            for (lr, lc, coeff) in v_coeffs:
                i = br + lr
                j = 4 + lc  # global col: 4 or 5
                b_idx = i * p + j
                V_col[b_idx] += coeff
            
            # W: C2[cr:cr+2, 0:2] -> global C[4:6, cr:cr+2]
            for (lr, lc, coeff) in w_coeffs:
                i = cr + lr
                k = 4 + lc  # global col: 4 or 5
                c_idx = k * n + i
                W_col[c_idx] += coeff
            
            U[:, offset + col] = U_col
            V[:, offset + col] = V_col
            W[:, offset + col] = W_col
            col += 1
        
        # Schoolbook for 3rd column (local col 2 of B2, global col 6)
        # C2[cr+lr, 2] = A[ar+lr, ac+0]*B2[br+0, 2] + A[ar+lr, ac+1]*B2[br+1, 2]
        for lr in range(2):
            for j in range(2):
                i = ar + lr
                j_a = ac + j
                j_b = br + j
                k = 6  # global col 6
                
                a_idx = i * m + j_a
                b_idx = j_b * p + k
                c_idx = k * n + i
                
                U[a_idx, offset + col] = 1.0
                V[b_idx, offset + col] = 1.0
                W[c_idx, offset + col] = 1.0
                col += 1
                
    # Validation (Optional but good for debugging)
    # We check a few random points to ensure correctness
    if time.time() - start_time < DEADLINE - 5:
        # Check target tensor entries
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # sum_r U[a,r] V[b,r] W[c,r] should be 1 for these, 0 otherwise.
        
        # Check a sample of target entries
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Validation failed for target entry ({a},{b},{c}): {val}")
        
        # Check a sample of non-target entries
        for _ in range(100):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if this is a target entry
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    raise ValueError(f"Validation failed for non-target entry ({a},{b},{c}): {val}")

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
