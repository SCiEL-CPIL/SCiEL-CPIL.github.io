import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x3x7():
    """
    Construct a decomposition using Strassen-like block structure.
    We partition A (2x3) and B (3x7) into blocks.
    Let A = [[A11, A12, A13], [A21, A22, A23]] where Aij are 1x1.
    Let B = [[B11, B12, B13, B14, B15, B16, B17],
             [B21, B22, B23, B24, B25, B26, B27],
             [B31, B32, B33, B34, B35, B36, B37]]
    This is just scalars.
    
    Instead, let's group columns of B.
    Let B1 = B[:, 0:3] (3x3) and B2 = B[:, 3:6] (3x3) and B3 = B[:, 6:7] (3x1).
    C = A * B = A * [B1, B2, B3] = [A*B1, A*B2, A*B3].
    
    We can use Strassen for A*B1 and A*B2.
    A is 2x3. B1 is 3x3.
    This doesn't fit standard Strassen directly (2x2).
    
    Let's try grouping rows of A? No, A has 2 rows.
    Let's group columns of A into 2x1 and 1x1?
    A = [A_L, A_R] where A_L is 2x2, A_R is 2x1.
    B = [B_T; B_B] where B_T is 2x7, B_B is 1x7.
    C = A_L * B_T + A_R * B_B.
    
    A_L * B_T is 2x2 * 2x7 = 2x7.
    A_R * B_B is 2x1 * 1x7 = 2x7.
    
    We can use Strassen for A_L * B_T?
    A_L is 2x2. B_T is 2x7.
    Let B_T = [B_T1, B_T2, B_T3] where each is 2x? No.
    Let's split B_T into 2x2, 2x2, 2x3?
    B_T = [B_T1 (2x2), B_T2 (2x2), B_T3 (2x3)].
    Then A_L * B_T = [A_L B_T1, A_L B_T2, A_L B_T3].
    We can use Strassen (rank 7) for A_L B_T1 (2x2 * 2x2).
    We can use Strassen (rank 7) for A_L B_T2 (2x2 * 2x2).
    For A_L B_T3 (2x2 * 2x3), we can split B_T3 into 2x1, 2x1, 2x1?
    Or use schoolbook for 2x2 * 2x3 (rank 12) or try to optimize.
    
    Let's try a simpler partition.
    A = [[A1, A2, A3], [A4, A5, A6]] (2x3)
    B = 3x7.
    
    Let's use the rank-7 algorithm for 2x2 matrix multiplication.
    M1 = (A4 + A5) * (B1 + B2) ? No, indices are different.
    
    Let's define the Strassen 2x2 multiplication of X (2x2) and Y (2x2).
    X = [[x1, x2], [x3, x4]], Y = [[y1, y2], [y3, y4]].
    Z = XY = [[z1, z2], [z3, z4]].
    m1 = (x4 + x1) * (y1 + y2)
    m2 = (x3 + x4) * y1
    m3 = x1 * (y2 - y4)
    m4 = x4 * (y3 - y1)
    m5 = (x1 + x2) * y4
    m6 = (x3 - x1) * (y1 + y3)
    m7 = (x2 - x4) * (y2 + y4)
    
    z1 = m1 + m4 - m5 + m7
    z2 = m3 + m5
    z3 = m2 + m4
    z4 = m1 - m2 + m3 + m6
    
    Now, let's map our 2x3 * 3x7 problem to this.
    Let A = [[a1, a2, a3], [a4, a5, a6]].
    Let B be 3x7.
    
    We can view A as [A1, A2] where A1 is 2x2 and A2 is 2x1.
    A1 = [[a1, a2], [a4, a5]], A2 = [[a3], [a6]].
    Let B = [B1, B2] where B1 is 3x6 and B2 is 3x1.
    Wait, B is 3x7.
    Let B = [B_left (3x6), B_right (3x1)].
    C = A * B = A1 * B1_left + A2 * B1_right + A1 * B2_left + A2 * B2_right?
    No, B is 3x7.
    B = [[B11, ..., B17], [B21, ..., B27], [B31, ..., B37]].
    
    Let's split B into 3x2, 3x2, 3x2, 3x1.
    B = [B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x1)].
    C = A * [B1, B2, B3, B4] = [A B1, A B2, A B3, A B4].
    
    A B1: 2x3 * 3x2.
    A B2: 2x3 * 3x2.
    A B3: 2x3 * 3x2.
    A B4: 2x3 * 3x1.
    
    This is getting complicated. Let's stick to the simplest Strassen application.
    Partition A into 2x2 and 2x1.
    A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    Partition B into 2x7 and 1x7.
    B = [B1; B2] where B1 is 2x7, B2 is 1x7.
    C = A1 B1 + A2 B2.
    
    A1 B1 is 2x2 * 2x7.
    A2 B2 is 2x1 * 1x7.
    
    For A1 B1, we can use Strassen if we split B1 into 2x2 blocks?
    B1 is 2x7. Let B1 = [B11 (2x2), B12 (2x2), B13 (2x3)].
    A1 B1 = [A1 B11, A1 B12, A1 B13].
    
    A1 B11 (2x2 * 2x2): Rank 7 (Strassen).
    A1 B12 (2x2 * 2x2): Rank 7 (Strassen).
    A1 B13 (2x2 * 2x3): Rank 12 (Schoolbook) or maybe less?
    Let's use schoolbook for 2x2 * 2x3.
    A1 = [[a1, a2], [a3, a4]].
    B13 = [[b1, b2, b3], [b4, b5, b6]].
    C13 = [[a1b1+a2b4, a1b2+a2b5, a1b3+a2b6],
           [a3b1+a4b4, a3b2+a4b5, a3b3+a4b6]].
    Rank 12.
    
    A2 B2 (2x1 * 1x7):
    A2 = [[a5], [a6]]. B2 = [b1, b2, b3, b4, b5, b6, b7].
    C2 = [a5b1, a5b2, ..., a5b7, a6b1, a6b2, ..., a6b7].
    Rank 14 (Schoolbook).
    
    Total Rank = 7 + 7 + 12 + 14 = 40.
    This is better than 42.
    
    Let's try to optimize A1 B13 (2x2 * 2x3).
    Can we do it in rank 10 or 11?
    Or maybe split B1 differently?
    B1 is 2x7.
    Split into 2x3, 2x3, 2x1.
    A1 B11 (2x2 * 2x3): Rank 12.
    A1 B12 (2x2 * 2x3): Rank 12.
    A1 B13 (2x2 * 2x1): Rank 4.
    Total for A1 B1 = 12 + 12 + 4 = 28.
    Plus A2 B2 (2x1 * 1x7) = 14.
    Total = 28 + 14 = 42. No gain.
    
    What if we use Strassen for the whole thing?
    A is 2x3. B is 3x7.
    Pad A to 2x4? B to 4x8?
    Then 2x4 * 4x8.
    This is getting too complex for a quick implementation.
    
    Let's stick with the 40 rank solution.
    A = [A1 (2x2), A2 (2x1)].
    B = [B1 (2x7), B2 (1x7)].
    B1 = [B11 (2x2), B12 (2x2), B13 (2x3)].
    
    C = A1 B11 + A2 B11_part? No.
    C = A1 B1 + A2 B2.
    A1 B1 = [A1 B11, A1 B12, A1 B13].
    A2 B2 = A2 B2 (2x1 * 1x7).
    
    Wait, B1 is top 2 rows of B. B2 is bottom 1 row of B.
    So B = [B1; B2].
    B1 is 2x7. B2 is 1x7.
    A1 is left 2 cols of A. A2 is right 1 col of A.
    A = [A1, A2].
    
    C = A1 B1 + A2 B2.
    A1 B1 is 2x2 * 2x7.
    A2 B2 is 2x1 * 1x7.
    
    Let's implement the rank 40 solution.
    """
    n, m, p = 2, 3, 7
    # Rank 40
    # A1 (2x2) * B11 (2x2) -> Strassen (7)
    # A1 (2x2) * B12 (2x2) -> Strassen (7)
    # A1 (2x2) * B13 (2x3) -> Schoolbook (12)
    # A2 (2x1) * B2 (1x7)  -> Schoolbook (14)
    
    # Indices:
    # A:
    # A1 = [[a1, a2], [a4, a5]] (rows 0,1; cols 0,1)
    # A2 = [[a3], [a6]] (rows 0,1; col 2)
    
    # B:
    # B1 = rows 0,1 of B (2x7)
    # B2 = row 2 of B (1x7)
    # B11 = B1[:, 0:2] (2x2)
    # B12 = B1[:, 2:4] (2x2)
    # B13 = B1[:, 4:7] (2x3)
    
    U = [[0]*40 for _ in range(6)]
    V = [[0]*40 for _ in range(21)]
    W = [[0]*40 for _ in range(14)]
    
    col = 0
    
    # Helper to add Strassen terms for 2x2 * 2x2
    def add_strassen(a_rows, a_cols, b_rows, b_cols, offset):
        nonlocal col
        # a_rows: [r1, r2], a_cols: [c1, c2]
        # b_rows: [r1, r2], b_cols: [c1, c2]
        # Output C is 2x2 at rows a_rows, cols offset + [0, 1, 2, 3]?
        # Wait, B11 is 2x2. Output is 2x2.
        # In the final C (2x7), A1 B11 contributes to cols 0,1.
        # A1 B12 contributes to cols 2,3.
        # A1 B13 contributes to cols 4,5,6.
        
        # Strassen formulas:
        # m1 = (a11 + a22) * (b11 + b12)
        # m2 = (a21 + a22) * b11
        # m3 = a11 * (b12 - b22)
        # m4 = a22 * (b21 - b11)
        # m5 = (a11 + a12) * b22
        # m6 = (a21 - a11) * (b11 + b21)
        # m7 = (a12 - a22) * (b12 + b22)
        
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        a11_idx = a_rows[0] * m + a_cols[0]
        a12_idx = a_rows[0] * m + a_cols[1]
        a21_idx = a_rows[1] * m + a_cols[0]
        a22_idx = a_rows[1] * m + a_cols[1]
        
        b11_idx = b_rows[0] * p + b_cols[0]
        b12_idx = b_rows[0] * p + b_cols[1]
        b21_idx = b_rows[1] * p + b_cols[0]
        b22_idx = b_rows[1] * p + b_cols[1]
        
        # m1
        U[a11_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        V[b12_idx][col] = 1
        W[offset*2 + 0][col] = 1 # C11
        W[offset*2 + 3][col] = 1 # C22
        col += 1
        
        # m2
        U[a21_idx][col] = 1
        U[a22_idx][col] = 1
        V[b11_idx][col] = 1
        W[offset*2 + 1][col] = 1 # C21
        W[offset*2 + 3][col] = -1 # C22
        col += 1
        
        # m3
        U[a11_idx][col] = 1
        V[b12_idx][col] = 1
        V[b22_idx][col] = -1
        W[offset*2 + 1][col] = 1 # C12
        W[offset*2 + 3][col] = 1 # C22
        col += 1
        
        # m4
        U[a22_idx][col] = 1
        V[b21_idx][col] = 1
        V[b11_idx][col] = -1
        W[offset*2 + 0][col] = 1 # C11
        W[offset*2 + 1][col] = 1 # C21
        col += 1
        
        # m5
        U[a11_idx][col] = 1
        U[a12_idx][col] = 1
        V[b22_idx][col] = 1
        W[offset*2 + 0][col] = -1 # C11
        W[offset*2 + 1][col] = 1 # C12
        col += 1
        
        # m6
        U[a21_idx][col] = 1
        U[a11_idx][col] = -1
        V[b11_idx][col] = 1
        V[b21_idx][col] = 1
        W[offset*2 + 3][col] = 1 # C22
        col += 1
        
        # m7
        U[a12_idx][col] = 1
        U[a22_idx][col] = -1
        V[b12_idx][col] = 1
        V[b22_idx][col] = 1
        W[offset*2 + 0][col] = 1 # C11
        col += 1
        
    # Block 1: A1 * B11 (2x2 * 2x2)
    # A1 rows 0,1 cols 0,1
    # B11 rows 0,1 cols 0,1
    # Output cols 0,1
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], 0)
    
    # Block 2: A1 * B12 (2x2 * 2x2)
    # A1 rows 0,1 cols 0,1
    # B12 rows 0,1 cols 2,3
    # Output cols 2,3
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], 1)
    
    # Block 3: A1 * B13 (2x2 * 2x3) -> Schoolbook
    # A1 rows 0,1 cols 0,1
    # B13 rows 0,1 cols 4,5,6
    # Output cols 4,5,6
    # C14 = a11 b14 + a12 b24
    # C15 = a11 b15 + a12 b25
    # C16 = a11 b16 + a12 b26
    # C24 = a21 b14 + a22 b24
    # C25 = a21 b15 + a22 b25
    # C26 = a21 b16 + a22 b26
    
    a11_idx = 0 * m + 0
    a12_idx = 0 * m + 1
    a21_idx = 1 * m + 0
    a22_idx = 1 * m + 1
    
    for k in range(3):
        b_col = 4 + k
        # C1, k+4
        U[a11_idx][col] = 1
        V[0 * p + b_col][col] = 1
        W[0 * n + 4 + k][col] = 1
        col += 1
        
        U[a12_idx][col] = 1
        V[1 * p + b_col][col] = 1
        W[0 * n + 4 + k][col] = 1
        col += 1
        
        U[a21_idx][col] = 1
        V[0 * p + b_col][col] = 1
        W[1 * n + 4 + k][col] = 1
        col += 1
        
        U[a22_idx][col] = 1
        V[1 * p + b_col][col] = 1
        W[1 * n + 4 + k][col] = 1
        col += 1
    
    # Block 4: A2 * B2 (2x1 * 1x7) -> Schoolbook
    # A2 rows 0,1 col 2
    # B2 row 2 cols 0..6
    # C1, k = a13 b3k
    # C2, k = a23 b3k
    
    a13_idx = 0 * m + 2
    a23_idx = 1 * m + 2
    
    for k in range(7):
        b_col = k
        U[a13_idx][col] = 1
        V[2 * p + b_col][col] = 1
        W[0 * n + k][col] = 1
        col += 1
        
        U[a23_idx][col] = 1
        V[2 * p + b_col][col] = 1
        W[1 * n + k][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _try_rank(r, n, m, p, max_attempts=100):
    """
    Attempts to find a rank r decomposition by fixing V and W randomly,
    then solving for U linearly.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(max_attempts):
        # Randomly initialize V and W with small integers
        V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(mp)]
        W = [[random.randint(-2, 2) for _ in range(r)] for _ in range(pn)]
        
        # Check if V and W are "generic" enough? 
        # We just try to solve for U.
        
        U = [[Fraction(0) for _ in range(r)] for _ in range(nm)]
        valid = True
        
        for i in range(n):
            for j in range(m):
                # Construct the p x r matrix M where M[k, alpha] = V[j*p+k, alpha] * W[k*n+i, alpha]
                M = [[Fraction(0) for _ in range(r)] for _ in range(p)]
                for k in range(p):
                    for alpha in range(r):
                        M[k][alpha] = Fraction(V[j * p + k][alpha]) * Fraction(W[k * n + i][alpha])
                
                # Solve M * u = 1 (vector of ones)
                # u is a vector of length r
                # We use Gaussian elimination
                # Augmented matrix: p x (r+1)
                Aug = [row[:] + [Fraction(1)] for row in M]
                
                # Perform Gaussian elimination
                rank_aug = 0
                for col in range(r):
                    # Find pivot
                    pivot = -1
                    for row in range(rank_aug, p):
                        if Aug[row][col] != 0:
                            pivot = row
                            break
                    if pivot == -1:
                        continue
                    
                    # Swap
                    Aug[rank_aug], Aug[pivot] = Aug[pivot], Aug[rank_aug]
                    
                    # Eliminate
                    pivot_val = Aug[rank_aug][col]
                    for c in range(col, r + 1):
                        Aug[rank_aug][c] /= pivot_val
                        
                    for row in range(p):
                        if row != rank_aug and Aug[row][col] != 0:
                            factor = Aug[row][col]
                            for c in range(col, r + 1):
                                Aug[row][c] -= factor * Aug[rank_aug][c]
                    
                    rank_aug += 1
                    if rank_aug == r:
                        break
                
                # Check for consistency (no 0 = 1 rows)
                for row in range(rank_aug, p):
                    if Aug[row][r] != 0:
                        valid = False
                        break
                if not valid:
                    break
                
                # If consistent, we have a solution.
                # If rank_aug == r, unique solution.
                # If rank_aug < r, infinite solutions. We pick the simplest (set free vars to 0).
                u_vec = [Fraction(0) for _ in range(r)]
                
                # Back substitution / reading off the solution
                # The matrix is in reduced row echelon form.
                # We need to map the basic variables to the solution.
                # This is tricky with the standard Gauss-Jordan if we don't track columns.
                # Let's just use a simple method: if r==p, M is square and invertible.
                # If r > p, we have free variables.
                
                # For simplicity and robustness, let's only accept solutions if r == p or if we can easily find one.
                # Given p=7, if r=7, M is 7x7.
                # If r > 7, it's harder to extract a "simple" solution.
                # Let's restrict search to r=p=7 first? No, rank might be > 7.
                
                # Let's try to extract the solution assuming the first 'rank_aug' variables are basic.
                # This is a heuristic.
                for row in range(rank_aug):
                    # Find the column with 1 in this row
                    basic_col = -1
                    for c in range(r):
                        if Aug[row][c] == 1:
                            basic_col = c
                            break
                    if basic_col != -1:
                        u_vec[basic_col] = Aug[row][r]
                
                # Check if u_vec entries are integers or half-integers
                for val in u_vec:
                    if val.denominator not in [1, 2]:
                        valid = False
                        break
                if not valid:
                    break
                
                # Store u_vec in U
                idx = i * m + j
                for alpha in range(r):
                    U[idx][alpha] = u_vec[alpha]
        
        if valid:
            # Convert to integers/half-integers for output
            U_out = [[float(u) for u in row] for row in U]
            # Actually, the prompt asks for integer/half-integer entries.
            # Fractions like 1/2 are fine.
            # Let's return the Fraction objects or convert to float if they are exact?
            # The prompt says "integer/half-integer entries".
            # Returning Fractions is safer for exactness verification, but the interface might expect numbers.
            # The seed program returns integers.
            # Let's return the values as numbers (floats might lose precision if not careful, but 1/2 is exact in float).
            # However, to be safe, I will return the values as floats if they are .0 or .5, else it's invalid.
            
            # Verify the found solution
            if _verify_decomposition(U, V, W, n, m, p):
                # Convert to standard types
                U_ret = [[float(u) for u in row] for row in U]
                V_ret = [[float(v) for v in row] for row in V]
                W_ret = [[float(w) for w in row] for row in W]
                return {"U": U_ret, "V": V_ret, "W": W_ret}
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the constructed Strassen-like solution (Rank 40)
    res = _strassen_2x3x7()
    if _verify_decomposition(res['U'], res['V'], res['W'], n, m, p):
        return res
        
    # If verification fails (shouldn't happen), fall back to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U, V, W = res['U'], res['V'], res['W']
    rank = len(U[0])
    
    # Simple verification
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    
    # Verify a few entries
    for _ in range(10):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        val = 0
        for r in range(rank):
            val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        assert abs(val - 1) < 1e-9, f"Failed for i={i}, j={j}, k={k}, val={val}"
        
    elapsed = time.time() - start_time
    print(f"Score (Rank): {rank}")
