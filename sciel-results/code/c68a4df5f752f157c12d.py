import time
import sys

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    # Strassen 2x2x2 patterns: (u_pattern, v_pattern, w_pattern)
    # Each pattern is a list of 4 values for the 2x2 block
    # u_pattern[i*2+j] corresponds to A[i,j]
    # v_pattern[j*2+k] corresponds to B[j,k]
    # w_pattern[k*2+i] corresponds to C[k,i] (trace convention W[k*n+i])
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    strassen_patterns = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Schoolbook 2x2x1 patterns (for the 5th column)
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    schoolbook_2x2x1 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),  # A[0,0]*B[0,0] -> C[0,0]
        ([0, 1, 0, 0], [0, 1], [1, 0]),  # A[0,1]*B[1,0] -> C[0,0]
        ([0, 0, 1, 0], [1, 0], [0, 1]),  # A[1,0]*B[0,0] -> C[1,0]
        ([0, 0, 0, 1], [0, 1], [0, 1]),  # A[1,1]*B[1,0] -> C[1,0]
    ]

    # Build the rank-36 decomposition
    # We split the 5 columns into 2+2+1
    # Block 1: columns 0,1 of B -> 2x2x2 Strassen (rank 7)
    # Block 2: columns 2,3 of B -> 2x2x2 Strassen (rank 7)
    # Block 3: column 4 of B -> 2x2x1 schoolbook (rank 4)
    # For each block, we need to handle both rows of A (n=2)
    # But A has 4 columns, so we split A into 2x2 + 2x1 blocks? No.
    # Actually, A is 2x4. We can view it as 2 rows, 4 columns.
    # For Strassen, we need 2x2 blocks of A.
    # So we split the 4 columns of A into two 2-column groups.
    # Block A_left: columns 0,1; Block A_right: columns 2,3
    
    # Let me reconsider the decomposition.
    # C = A * B where A is 2x4, B is 4x5, C is 2x5
    # Split A = [A1 | A2] where A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3)
    # Split B = [[B11; B21] | [B12; B22] | [B13; B23]]
    #   B11 is 2x2 (rows 0,1; cols 0,1)
    #   B21 is 2x2 (rows 2,3; cols 0,1)
    #   B12 is 2x2 (rows 0,1; cols 2,3)
    #   B22 is 2x2 (rows 2,3; cols 2,3)
    #   B13 is 2x1 (rows 0,1; col 4)
    #   B23 is 2x1 (rows 2,3; col 4)
    
    # C[:, 0:2] = A1 * B11 + A2 * B21
    # C[:, 2:4] = A1 * B12 + A2 * B22
    # C[:, 4]   = A1 * B13 + A2 * B23
    
    # For C[:, 0:2] (2x2 output):
    #   This is A1*B11 + A2*B21, each product is 2x2.
    #   We can use Strassen on A1*B11 (rank 7) and A2*B21 (rank 7) = rank 14
    #   But we can potentially share? No, they're independent.
    #   Actually, we can combine: let D1 = A1+A2, E1 = B11-B21, etc.
    #   But the standard approach: just do two Strassen blocks = 14
    
    # Similarly for C[:, 2:4]: A1*B12 + A2*B22 = 14
    # For C[:, 4]: A1*B13 + A2*B23 = 4 + 4 = 8? No, each is 2x1.
    #   A1*B13: 2x2 * 2x1 = 2x1, rank 4 schoolbook
    #   A2*B23: 2x2 * 2x1 = 2x1, rank 4 schoolbook
    #   Total: 4 + 4 = 8? But we can do better.
    #   Actually 2x2 * 2x1 has rank 4 (schoolbook). Can we do better?
    #   No, 2x2x1 tensor has rank 4.
    
    # Total: 14 + 14 + 8 = 36. That matches.
    
    # But wait, can we do C[:, 0:2] more efficiently?
    # A1*B11 + A2*B21 where all are 2x2.
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
    # The rank of 2x4x2 matrix multiplication... 
    # I believe it's 14 (two Strassen blocks).
    
    # Let's just implement it as:
    # For output cols 0,1: Strassen(A1, B11) + Strassen(A2, B21) = 14
    # For output cols 2,3: Strassen(A1, B12) + Strassen(A2, B22) = 14
    # For output col 4: schoolbook(A1, B13) + schoolbook(A2, B23) = 8
    # Total = 36

    rank_total = 36
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]

    def add_strassen_block(u_i_off, u_j_off, v_j_off, v_k_off, w_i_off, w_k_off, col_off):
        """Add a 2x2x2 Strassen block.
        u_i_off: row offset for A (0 or 1)
        u_j_off: col offset for A (0, 2)
        v_j_off: row offset for B (0, 2)
        v_k_off: col offset for B (0, 2, 4)
        w_i_off: row offset for C (0 or 1) -- but C is 2x5, so i in 0,1
        w_k_off: col offset for C (0, 2, 4)
        """
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = u_i_off + i_local
                    j_global = u_j_off + j_local
                    U[i_global * m + j_global][col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = v_j_off + j_local
                    k_global = v_k_off + k_local
                    V[j_global * p + k_global][col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = w_i_off + i_local
                    k_global = w_k_off + k_local
                    W[k_global * n + i_global][col_off] = val
            col_off += 1
        return col_off

    def add_schoolbook_2x2x1(u_i_off, u_j_off, v_j_off, v_k_off, w_i_off, w_k_off, col_off):
        """Add a 2x2x1 schoolbook block (for single output column)."""
        for u_pat, v_pat, w_pat in schoolbook_2x2x1:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = u_i_off + i_local
                    j_global = u_j_off + j_local
                    U[i_global * m + j_global][col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = v_j_off + j_local
                    k_global = v_k_off + k_local
                    V[j_global * p + k_global][col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = w_i_off + i_local
                    k_global = w_k_off + k_local
                    W[k_global * n + i_global][col_off] = val
            col_off += 1
        return col_off

    col = 0

    # Output cols 0,1: from A cols 0,1 times B cols 0,1 + A cols 2,3 times B cols 0,1
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)      # A[0:2,0:2] * B[0:2,0:2]
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)      # A[0:2,2:4] * B[2:4,0:2]

    # Output cols 2,3: from A cols 0,1 times B cols 2,3 + A cols 2,3 times B cols 2,3
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)      # A[0:2,0:2] * B[0:2,2:4]
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)      # A[0:2,2:4] * B[2:4,2:4]

    # Output col 4: from A cols 0,1 times B col 4 + A cols 2,3 times B col 4
    col = add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4, col)    # A[0:2,0:2] * B[0:2,4]
    col = add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4, col)    # A[0:2,2:4] * B[2:4,4]

    assert col == rank_total, f"Expected {rank_total}, got {col}"

    # Verify
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank_total):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check cross terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = 0
                    for r in range(rank_total):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify():
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 40
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(nm)]
        V_sb = [[0] * rank_sb for _ in range(mp)]
        W_sb = [[0] * rank_sb for _ in range(pn)]
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col2] = 1
                    V_sb[j * p + k][col2] = 1
                    W_sb[k * n + i][col2] = 1
                    col2 += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
