"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-28 decomposition exploiting rectangular structure via Strassen on 2x2 blocks
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 15 rows
    W = [[0] * rank for _ in range(p * n)]  # 10 rows

    # Strassen for 2x2 matrices:
    # A = [[a, b], [d, e]], B = [[g, h], [k, l]]
    # M1 = (a+e)(g+l)
    # M2 = (d+e)g
    # M3 = a(h-l)
    # M4 = e(k-g)
    # M5 = (a+b)l
    # M6 = (d-a)(g+h)
    # M7 = (b-e)(k+l)
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7

    def add_strassen_block(k_offset, col_offset):
        """Add Strassen decomposition for A(:, k_offset:k_offset+2) * B(k_offset:k_offset+2, :)
           A columns 0,1,2 -> rows 0,1,2 of U
           B rows 0,1,2 -> rows k_offset*5..(k_offset+1)*5 of V  (but only 2 rows used)
        """
        # For 2x3 * 2x5 subproblem:
        # A = [[a,b,c],[d,e,f]], B = [[g,h,i,j,k],[l,m,o,p,q]]
        # We split A into A1 = [[a,b],[d,e]] (2x2) and A2 = [[c],[f]] (2x1)
        # B into B1 = [[g,h],[l,m]] (2x2) and B2 = [[i,j,k],[o,p,q]] (2x3)
        # C = A1*B1 + A2*B2
        # A1*B1 uses 7 mults (Strassen)
        # A2*B2 uses 6 mults (schoolbook: c*i, c*j, c*k, f*o, f*p, f*q)
        # Total: 13 mults per 2-column block? No wait.
        
        # Actually let me reconsider. The 2x3 * 3x5 problem:
        # Split B into 3 column blocks of 2,2,1? No, p=5 columns of B.
        # A is 2x3, B is 3x5.
        # Split along the inner dimension m=3.
        # B has 3 rows. Split B into B1 (2 rows) and B2 (1 row).
        # A has 3 columns. Split A into A1 (2 cols) and A2 (1 col).
        # C = A1*B1 + A2*B2
        # A1 is 2x2, B1 is 2x5: rank of 2x2*2x5 = ?
        # A2 is 2x1, B2 is 1x5: rank of 2x1*1x5 = 10 (schoolbook)
        
        # Hmm, that gives 2x2*2x5 + 2x1*1x5.
        # 2x2*2x5: can use Strassen on the 2x2 part? 
        # A1 = [[a,b],[d,e]], B1 = [[g,h,i,j,k],[l,m,o,p,q]]
        # A1*B1 = 2x5 matrix.
        # This is like multiplying 2x2 by 2x5.
        # We can think of it as 5 separate 2x2*1 problems? No.
        # Actually, 2x2 * 2x5 has tensor rank. Let me think differently.
        
        # Let me just use the approach from the knowledge:
        # Decompose M_{2,3,5} by splitting the 3 columns of A and 3 rows of B.
        # Split into: A1 (2 cols) * B1 (2 rows) + A2 (1 col) * B2 (1 row)
        # A1*B1: 2x2 * 2x5 -> this is a 2x5 output from two 2x2 "slices"
        # Actually, let me just implement the known rank-28 directly.
        
        pass

    # Let me implement it more carefully.
    # The multiplication (A B) where A is 2x3, B is 3x5.
    # A = [A_left (2x2) | A_right (2x1)]
    # B = [B_top (2x5); B_bottom (1x5)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 times 2x5
    # Let A_left = [[a, b], [d, e]]
    # B_top = [[b0, b1, b2, b3, b4], [c0, c1, c2, c3, c4]]
    # Result is 2x5.
    # We can write this as: for each column j of B_top, compute A_left * [b_j; c_j]
    # That's 5 separate 2x2 * 2x1 multiplications, each of rank 4 (schoolbook) or 
    # we can use a shared decomposition.
    
    # Actually, the tensor for 2x2 * 2x5:
    # The 2x2 part uses Strassen (7 mults) but applied to produce a 2x5 output.
    # Hmm, this is getting complicated. Let me just use the block decomposition:
    # Split the 5 columns of B into blocks: 2, 2, 1
    # For each 2-column block of B (2 rows x 2 cols), we have a 2x2 * 2x2 subproblem.
    # For the 1-column block, we have a 2x2 * 2x1 subproblem.
    
    # 2x2 * 2x2 via Strassen: rank 7
    # 2x2 * 2x1 via Strassen-like: rank 4 (schoolbook) or less?
    # 2x2 * 2x1: A=[[a,b],[d,e]], B=[[x],[y]], C=[[ax+by],[dx+ey]]
    # This has rank 4 (schoolbook). Can we do better? It's just 4 terms, rank 4.
    
    # So: 7 + 7 + 4 = 18 for A_left * B_top? No wait.
    # A_left is 2x2, B_top is 2x5.
    # Split B_top into B_top1 (2x2), B_top2 (2x2), B_top3 (2x1).
    # A_left * B_top1: 2x2*2x2, rank 7
    # A_left * B_top2: 2x2*2x2, rank 7  
    # A_left * B_top3: 2x2*2x1, rank 4
    # Total for A_left * B_top: 7 + 7 + 4 = 18
    
    # A_right * B_bottom: 2x1 * 1x5
    # A_right = [[c],[f]], B_bottom = [g, h, i, j, k]
    # Result = [[c*g, c*h, c*i, c*j, c*k], [f*g, f*h, f*i, f*j, f*k]]
    # Rank: 5 (each of c*g, c*h, ... is one, but c and f are shared)
    # Actually this is a rank-5 tensor: outer product of [c,f] and [g,h,i,j,k]
    # So rank 5.
    
    # Total: 18 + 5 = 23? That seems too good. Let me double check.
    # Wait, I think I'm confusing myself. Let me restart with a cleaner approach.
    
    # Direct approach: use the known result that R(M_{2,3,5}) <= 28.
    # The decomposition from the knowledge base uses:
    # Split the 3x3 inner dimension into 2+1.
    # For the 2-part: use Strassen (rank 7) for each 2x2 block of the 5 columns
    # For the 1-part: schoolbook
    
    # Let me just hardcode the rank-28 decomposition.
    
    # U has 6 rows (n*m = 2*3), V has 15 rows (m*p = 3*5), W has 10 rows (p*n = 5*2)
    
    # I'll construct it as follows:
    # Columns 0-6: Strassen for A[:,0:2] * B[0:2, :]  (but this needs care)
    # Actually, let me just use a simpler known construction.
    
    # Schoolbook is rank 30. Let me implement rank 28 as described in the knowledge.
    
    # The key insight: for the 2x3 * 3x5 problem, split the middle dimension 3 = 2+1.
    # Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5)
    # Part 2: A[:, 2:3] (2x1) * B[2:3, :] (1x5)
    
    # Part 1: 2x2 * 2x5
    # Decompose B[0:2, :] into three 2x? blocks: 2x2, 2x2, 2x1
    # 2x2 * 2x2 via Strassen: 7 mults
    # 2x2 * 2x2 via Strassen: 7 mults  
    # 2x2 * 2x1: 4 mults (schoolbook)
    # Subtotal: 18
    
    # Part 2: 2x1 * 1x5 = 5 mults (outer product)
    # Subtotal: 5
    
    # Total: 23. But the knowledge says 28. I must be making an error.
    
    # Ah, I see the issue. The 2x2 * 2x5 cannot be decomposed that way because
    # the Strassen decomposition of 2x2*2x2 doesn't directly apply to 2x2*2x5.
    # Each "column pair" of the 2x5 needs its own set of Strassen mults.
    
    # So 2x2 * 2x5: treat as two 2x2*2x2 (using cols 0-1 and 2-3) and one 2x2*2x1 (col 4)
    # That's 7 + 7 + 4 = 18. This should be valid.
    
    # And 2x1 * 1x5: this is rank 5.
    
    # Total 23. If this works, it's better than 28!
    
    # Let me verify the 2x2*2x1 rank. 
    # A=[[a,b],[d,e]], B=[[x],[y]], C=[[ax+by],[dx+ey]]
    # Terms: ax, by, dx, ey. That's 4 terms, rank 4. Correct.
    
    # And 2x1*1x5: A=[a; d], B=[b0,b1,b2,b3,b4]
    # C = [[a*b0, a*b1, a*b2, a*b3, a*b4], [d*b0, d*b1, d*b2, d*b3, d*b4]]
    # This is the outer product of [a,d] and [b0,b1,b2,b3,b4], rank 5. Correct.
    
    # So total rank = 7+7+4+5 = 23. Let me implement this!
    
    rank = 23
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(15)]
    W = [[0] * rank for _ in range(10)]
    
    col = 0
    # Strassen for 2x2 * 2x2, columns 0-1 of B
    # A_left = [[a,b],[d,e]] corresponds to U rows: 0(a),1(b),3(d),4(e)
    # B_top cols 0-1 = [[b0,b1],[c0,c1]] corresponds to V rows: 0(b0),1(c0),2(b1),3(c1)
    # Output: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    # W rows: 0(C[0,0]),2(C[0,1]),4(C[1,0]),6(C[1,1])
    
    # Strassen mults:
    # M1=(a+e)(b0+c1)
    # M2=(d+e)b0
    # M3=a(c1-b0)
    # M4=e(b1-b0)  # wait, standard Strassen: M4=e(B21-B11)=e(c0-b0)
    # Let me use standard Strassen carefully.
    # A=[[a,b],[d,e]], B=[[g,h],[k,l]]
    # M1=(a+e)(g+l), M2=(d+e)g, M3=a(h-l), M4=e(k-g), M5=(a+b)l, M6=(d-a)(g+h), M7=(b-e)(k+l)
    # C11=M1-M4+M6, C12=M3+M5, C21=M2+M4, C22=M1-M3+M7
    
    # For first 2x2 block (B cols 0,1):
    # g=b0(V row 0), h=b1(V row 2), k=c0(V row 1), l=c1(V row 3)
    # a=U[0], b=U[1], d=U[3], e=U[4]
    
    # M1
    for r in [0, 4]:
        U[r][col] = 1
    for r in [0, 3]:
        V[r][col] = 1
    W[0][col] = 1; W[6][col] = 1
    col += 1
    # M2
    for r in [3, 4]:
        U[r][col] = 1
    V[0][col] = 1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    # M3
    U[0][col] = 1
    for r in [2, 1]:
        V[r][col] = 1
    W[2][col] = 1; W[6][col] = -1
    col += 1
    # M4
    U[4][col] = 1
    for r in [1, 0]:
        V[r][col] = 1
    W[4][col] = 1; W[0][col] = -1
    col += 1
    # M5
    for r in [0, 1]:
        U[r][col] = 1
    V[3][col] = 1
    W[2][col] = 1
    col += 1
    # M6
    for r in [3, 0]:
        U[r][col] = 1
        V[r + 2][col] = 1
    W[0][col] = 1
    col += 1
    # M7
    for r in [1, 4]:
        U[r][col] = 1
    for r in [1, 3]:
        V[r][col] = 1
    W[6][col] = 1
    col += 1
    
    # Second 2x2 block (B cols 2,3):
    # g=b2(V row 4), h=b3(V row 6), k=c2(V row 5), l=c3(V row 7)
    # W rows: 1(C[0,2]),3(C[0,3]),5(C[1,2]),7(C[1,3])
    
    # M1
    for r in [0, 4]:
        U[r][col] = 1
    for r in [4, 7]:
        V[r][col] = 1
    W[1][col] = 1; W[7][col] = 1
    col += 1
    # M2
    for r in [3, 4]:
        U[r][col] = 1
    V[4][col] = 1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    # M3
    U[0][col] = 1
    for r in [6, 5]:
        V[r][col] = 1
    W[3][col] = 1; W[7][col] = -1
    col += 1
    # M4
    U[4][col] = 1
    for r in [5, 4]:
        V[r][col] = 1
    W[5][col] = 1; W[1][col] = -1
    col += 1
    # M5
    for r in [0, 1]:
        U[r][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    # M6
    for r in [3, 0]:
        U[r][col] = 1
        V[r + 4][col] = 1
    W[1][col] = 1
    col += 1
    # M7
    for r in [1, 4]:
        U[r][col] = 1
    for r in [5, 7]:
        V[r][col] = 1
    W[7][col] = 1
    col += 1
    
    # 2x2 * 2x1 block (B col 4):
    # g=b4(V row 8), k=c4(V row 9)
    # C[0,4] = a*b4 + b*c4 (W row 2)
    # C[1,4] = d*b4 + e*c4 (W row 6)
    
    # a*b4
    U[0][col] = 1; V[8][col] = 1; W[2][col] = 1
    col += 1
    # b*c4
    U[1][col] = 1; V[9][col] = 1; W[2][col] = 1
    col += 1
    # d*b4
    U[3][col] = 1; V[8][col] = 1; W[6][col] = 1
    col += 1
    # e*c4
    U[4][col] = 1; V[9][col] = 1; W[6][col] = 1
    col += 1
    
    # Part 2: A[:,2] * B[2,:]
    # A[:,2] = [[c],[f]] -> U rows 2(c), 5(f)
    # B[2,:] = [g,h,i,j,k] -> V rows 10,11,12,13,14
    # Output: C[0,*] = c * [g,h,i,j,k] -> W rows 0,1,2,3,4
    # C[1,*] = f * [g,h,i,j,k] -> W rows 5,6,7,8,9
    
    for j in range(5):
        U[2][col] = 1; V[10 + j][col] = 1; W[j][col] = 1
        col += 1
        U[5][col] = 1; V[10 + j][col] = 1; W[5 + j][col] = 1
        col += 1
    
    assert col == 23, f"Expected 23, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
