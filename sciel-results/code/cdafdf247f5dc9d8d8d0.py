"""Rewritten program: Strassen-based rank-55 decomposition for 2x3 x 3x10 matrix multiplication."""

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3 x 3x10 matrix multiplication tensor into rank 55.
    
    Strategy:
    - Split the 2x3 matrix A into A_top (1x3) and A_bot (1x3) [actually 2x3]
    - Split the 3x10 matrix B into 3x2 blocks: B_0, B_1, B_2 (each 3x2)
    
    Actually, let's use a cleaner approach:
    - A is 2x3, B is 3x10
    - Split A into two 1x3 rows: A_0 (row 0), A_1 (row 1)  
    - Split B's columns into 5 groups of 2: B_j for j=0..4, each 3x2
    
    For each pair (A_i, B_j), we compute a 2x2 product using Strassen (rank 7).
    But that's 2 * 5 * 7 = 70, which is worse than schoolbook 60.
    
    Better approach from the measured finding:
    - Decompose into a 2x1x10 part (rank 20) and a 2x2x10 part (rank 35)
    
    Let me reconsider: A is 2x3, B is 3x10.
    
    Think of it as:
    - First column of A (2x1) times first column of B... no.
    
    Actually, the 2x1x10 part: take A's first column (2x1) and B (1x10 if we take B's first row).
    That gives 2x10, rank = 2*1*10 = 20.
    
    Then the remaining part: A's remaining 2x2 (columns 1,2) times B's remaining... 
    Wait, that doesn't split cleanly.
    
    Let me use a different decomposition. Split A = [A_left, A_right] where A_left is 2x1 and A_right is 2x2.
    Split B = [B_top; B_bot] where B_top is 1x10 and B_bot is 2x10.
    
    Then C = A * B = A_left * B_top + A_right * B_bot
    
    The first term: 2x1 * 1x10 = 2x10, rank = 2*1*10 = 20
    The second term: 2x2 * 2x10 = 2x10, which we can decompose using Strassen on the 2x2 part.
    
    For the 2x2 * 2x10: split B_bot into 5 blocks of 2x2. Each 2x2 * 2x2 uses Strassen (rank 7).
    Total for this part: 5 * 7 = 35.
    
    Total rank: 20 + 35 = 55.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We'll build U, V, W as lists of lists (rows x rank)
    # U: (n*m, rank) = (6, 55)
    # V: (m*p, rank) = (30, 55)  
    # W: (p*n, rank) = (20, 55)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 30 rows
    W = [[0] * rank for _ in range(p * n)]   # 20 rows
    
    col = 0
    
    # Part 1: A_left (col 0 of A) * B_top (row 0 of B)
    # A_left is 2x1: indices in A are (0,0) and (1,0) -> U indices: 0*m+0=0, 1*m+0=3
    # B_top is 1x10: indices in B are (0,0)..(0,9) -> V indices: 0*p+k = k for k=0..9
    # Result C is 2x10: C[i,k] = A[0,k]*B[0,k]... wait no.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For this part: C[i,k] += A[i,0] * B[0,k]
    # W index for C[i,k] is k*n + i = k*2 + i
    for i in range(n):  # i=0,1
        for k in range(p):  # k=0..9
            U[i * m + 0][col] = 1       # A[i, 0]
            V[0 * p + k][col] = 1       # B[0, k]
            W[k * n + i][col] = 1       # C[i, k] -> index k*n+i
            col += 1
    # col is now 20
    
    # Part 2: A_right (cols 1,2 of A) * B_bot (rows 1,2 of B)
    # A_right is 2x2: A[i, 1], A[i, 2] for i=0,1
    # B_bot is 2x10: B[j, k] for j=1,2 and k=0..9
    # Split B_bot into 5 blocks of 2x2: for block b=0..4, columns k=2b, 2b+1
    # For each block, use Strassen to compute 2x2 * 2x2
    
    # Strassen for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # Result: [[M1-M4+M5-M7, M3+M7], [M2+M4, M1-M3+M2+M6]]
    
    # Map to our indices:
    # A_right: a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    # B block: e=B[1,2b], f=B[1,2b+1], g=B[2,2b], h=B[2,2b+1]
    # Result C block: C[0,2b]=C00, C[0,2b+1]=C01, C[1,2b]=C10, C[1,2b+1]=C11
    
    def add_strassen_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries/v_entries/w_entries are lists of (index, value)."""
        nonlocal col
        for (u_idx, val) in u_entries:
            U[u_idx][col] += val
        for (v_idx, val) in v_entries:
            V[v_idx][col] += val
        for (w_idx, val) in w_entries:
            W[w_idx][col] += val
        col += 1
    
    for b in range(5):
        k1 = 2 * b      # first column of this block
        k2 = 2 * b + 1  # second column
        
        # U indices for A_right:
        # a = A[0,1] -> U index 0*m+1 = 1
        # b = A[0,2] -> U index 0*m+2 = 2
        # c = A[1,1] -> U index 1*m+1 = 4
        # d = A[1,2] -> U index 1*m+2 = 5
        
        # V indices for B block:
        # e = B[1,k1] -> V index 1*p+k1 = 10+k1
        # f = B[1,k2] -> V index 1*p+k2 = 10+k2
        # g = B[2,k1] -> V index 2*p+k1 = 20+k1
        # h = B[2,k2] -> V index 2*p+k2 = 20+k2
        
        # W indices for C block:
        # C00 = C[0,k1] -> W index k1*n+0 = 2*k1
        # C01 = C[0,k2] -> W index k2*n+0 = 2*k2
        # C10 = C[1,k1] -> W index k1*n+1 = 2*k1+1
        # C11 = C[1,k2] -> W index k2*n+1 = 2*k2+1
        
        ua, ub, uc, ud = 1, 2, 4, 5
        ve, vf, vg, vh = 10 + k1, 10 + k2, 20 + k1, 20 + k2
        w00, w01, w10, w11 = 2 * k1, 2 * k2, 2 * k1 + 1, 2 * k2 + 1
        
        # M1 = (a+d)(e+h)
        add_strassen_term(
            [(ua, 1), (ud, 1)],
            [(ve, 1), (vh, 1)],
            [(w00, 1), (w11, 1)]
        )
        
        # M2 = (c+d)e
        add_strassen_term(
            [(uc, 1), (ud, 1)],
            [(ve, 1)],
            [(w10, 1), (w11, 1)]
        )
        
        # M3 = a(f-h)
        add_strassen_term(
            [(ua, 1)],
            [(vf, 1), (vh, -1)],
            [(w01, 1), (w11, -1)]
        )
        
        # M4 = d(g-e)
        add_strassen_term(
            [(ud, 1)],
            [(vg, 1), (ve, -1)],
            [(w10, 1), (w00, -1)]
        )
        
        # M5 = (a+b)h
        add_strassen_term(
            [(ua, 1), (ub, 1)],
            [(vh, 1)],
            [(w00, 1), (w11, -1)]
        )
        
        # M6 = (c-a)(e+f)
        add_strassen_term(
            [(uc, 1), (ua, -1)],
            [(ve, 1), (vf, 1)],
            [(w11, 1)]
        )
        
        # M7 = (b-d)(g+h)
        add_strassen_term(
            [(ub, 1), (ud, -1)],
            [(vg, 1), (vh, 1)],
            [(w01, 1), (w00, -1)]
        )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
