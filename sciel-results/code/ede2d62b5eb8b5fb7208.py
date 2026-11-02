"""Matrix multiplication 2x3x10 rank decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    # U: n*m = 6 rows, rank cols
    # V: m*p = 30 rows, rank cols
    # W: p*n = 20 rows, rank cols
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen decomposition for each 2x2 block of B (3 blocks, cols 0-1, 2-3, 4-5)
    # For block r (r = 0, 1, 2), we use 7 scalar multiplications.
    # A = [[a, b], [c, d]]  (2x3: rows 0,1; cols 0,1,2)
    # B_block = [[e, f], [g, h]]  (2x2)
    # C = A * B_block (2x2)
    # P1 = a * f
    # P2 = (a+b) * h
    # P3 = a * (e-f)
    # P4 = d * (g-h)
    # P5 = (a+d) * (e+h)
    # P6 = (c-d) * (e+f)
    # P7 = (b-c) * (g+h)
    # C11 = P5 + P4 - P2 + P6
    # C12 = P1 + P2 - P5 - P7
    # C21 = P3 + P5 - P1 - P4
    # C22 = P1 + P3 - P2 + P7
    #
    # But we also need to handle the third column of A (column 2 of the 2x3 matrix).
    # A = [[a, b, c3], [c, d, d3]] where a=A[0,0], b=A[0,1], c3=A[0,2],
    #       c=A[1,0], d=A[1,1], d3=A[1,2]
    #
    # Actually, let me reconsider the block structure.
    # A is 2x3, B is 3x10.
    # C = A * B is 2x10.
    #
    # Split B into 5 blocks of 3x2: B_r for r=0..4, where B_r has columns 2r, 2r+1.
    # C_r = A * B_r is 2x2.
    #
    # For each block r, A is 2x3 and B_r is 3x2.
    # We can use a decomposition for 2x3 * 3x2 multiplication.
    #
    # Let's use the approach: split A into columns a0, a1, a2 (each 2x1).
    # B_r into rows b0, b1, b2 (each 1x2).
    # C_r = a0*b0 + a1*b1 + a2*b2 (each term is 2x2).
    #
    # We can use Strassen-like approach on the 2x2 outer products.
    # For a0*b0: a0 = [a, c]^T, b0 = [e, f] (row vector).
    # a0*b0 = [[a*e, a*f], [c*e, c*f]]
    # This is 4 scalar products, but we can do better.
    #
    # Actually, for 2x1 * 1x2 = 2x2, the rank is 4 (can't do better than schoolbook
    # for rank 2x2 matrices in general, unless we use structure).
    #
    # Let me think differently. For each 3x2 block of B, we multiply A (2x3) by it.
    # The product is 2x2. We can use a rank-5 or rank-6 decomposition for 2x3 * 3x2.
    #
    # Schoolbook for 2x3 * 3x2: 6 scalar products (2*3*2/2 = 6... no, 2*3*2=12, but
    # each scalar product gives one entry... actually 2*3=6 products for each of 2 columns = 12).
    # Wait: C[i][j] = sum_k A[i][k] * B[k][j]. For 2x3*3x2, there are 2*2=4 entries,
    # each requiring 3 products, so 12 products total.
    #
    # But we can share: C[i][0] = A[i][0]*B[0][0] + A[i][1]*B[1][0] + A[i][2]*B[2][0]
    # C[i][1] = A[i][0]*B[0][1] + A[i][1]*B[1][1] + A[i][2]*B[2][1]
    #
    # For fixed i, we need A[i][0], A[i][1], A[i][2] (3 values) to multiply with
    # B[0][0], B[1][0], B[2][0] and B[0][1], B[1][1], B[2][1].
    #
    # This is like multiplying a 1x3 vector by two 3x1 vectors... 
    #
    # Let me use a different approach. For the full 2x3 * 3x10, I'll use the fact
    # that we can decompose using 5 blocks of Strassen for 2x2 submatrices.
    #
    # Actually, the knowledge says: "Replacing trivial rank-60 decomposition with a
    # block-wise Strassen+schoolbook decomposition (5 blocks of rank 11 each) reduces
    # rank to 55."
    #
    # So for each of the 5 blocks (3x2 submatrix of B), we use rank 11.
    # Let me figure out the rank-11 decomposition for 2x3 * 3x2.
    #
    # 2x3 * 3x2 = 2x2. Schoolbook: 12 products.
    # We need rank 11, so save 1 product.
    #
    # Let A = [[a, b, c], [d, e, f]] (2x3)
    # B = [[g, h], [i, j], [k, l]] (3x2)
    # C = [[a*g+b*i+c*k, a*h+b*j+c*l],
    #      [d*g+e*i+f*k, d*h+e*j+f*l]]
    #
    # C[0][0] = a*g + b*i + c*k
    # C[0][1] = a*h + b*j + c*l
    # C[1][0] = d*g + e*i + f*k
    # C[1][1] = d*h + e*j + f*l
    #
    # 12 terms. We need to save 1.
    #
    # Idea: Let P1 = a*(g+h), P2 = a*(h-j), P3 = b*(i+j), P4 = b*(j-l),
    # P5 = c*(k+l), P6 = c*(l-m)... this is getting complicated.
    #
    # Let me try: 
    # P1 = a*g
    # P2 = a*h
    # P3 = b*i
    # P4 = b*j
    # P5 = c*k
    # P6 = c*l
    # P7 = d*g
    # P8 = d*h
    # P9 = e*i
    # P10 = e*j
    # P11 = f*k
    # P12 = f*l
    # That's 12.
    #
    # To save 1: notice that C[0][1] - C[0][0] = a*(h-g) + b*(j-i) + c*(l-k)
    # C[1][1] - C[1][0] = d*(h-g) + e*(j-i) + f*(l-k)
    #
    # Let u = h-g, v = j-i, w = l-k
    # C[0][0] = a*g + b*i + c*k
    # C[0][1] = C[0][0] + a*u + b*v + c*w
    # C[1][0] = d*g + e*i + f*k
    # C[1][1] = C[1][0] + d*u + e*v + f*w
    #
    # So: P1 = a*g, P2 = a*u, P3 = b*i, P4 = b*v, P5 = c*k, P6 = c*w,
    # P7 = d*g, P8 = d*u, P9 = e*i, P10 = e*v, P11 = f*k, P12 = f*w
    # Still 12.
    #
    # Hmm. Let me try another approach.
    # C[0][0] = a*g + b*i + c*k
    # C[1][1] = d*h + e*j + f*l
    # C[0][1] - C[1][1] = (a-d)*h + (b-e)*j + (c-f)*l
    # C[1][0] - C[0][0] = (d-a)*g + (e-b)*i + (f-c)*k
    #
    # Let P1 = a*g, P2 = b*i, P3 = c*k, P4 = d*h, P5 = e*j, P6 = f*l,
    # P7 = (a-d)*h, P8 = (b-e)*j, P9 = (c-f)*l, P10 = (d-a)*g, P11 = (e-b)*i, P12 = (f-c)*k
    # C[0][0] = P1 + P2 + P3
    # C[1][1] = P4 + P5 + P6
    # C[0][1] = P4 + P5 + P6 + P7 + P8 + P9
    # C[1][0] = P1 + P2 + P3 + P10 + P11 + P12
    # Still 12.
    #
    # For rank 11, I need a cleverer decomposition. Let me just use rank 12 per block
    # for 60 total, or find the rank-11.
    #
    # Actually, let me just use the simple approach: for each block, use schoolbook
    # but share some terms. The known result is rank 55 = 5*11.
    #
    # Let me try: for 2x3 * 3x2:
    # P1 = a*g
    # P2 = a*(h-j)  => a*h - a*j
    # P3 = b*i
    # P4 = b*(j-l)  => b*j - b*l
    # P5 = c*k
    # P6 = c*l
    # P7 = d*g
    # P8 = d*h
    # P9 = e*i
    # P10 = e*j
    # P11 = f*k
    # P12 = f*(l-k) => f*l - f*k... 
    #
    # C[0][0] = P1 + P3 + P5
    # C[0][1] = P1 + P2 + P3 + P4 + P5 + P6
    # C[1][0] = P7 + P9 + P11
    # C[1][1] = P7 + P8 + P9 + P10 + P11 + P12
    # That's 12.
    #
    # Let me try rank 11:
    # P1 = a*g
    # P2 = (a+b)*(h+i)  -- cross terms, messy
    #
    # I'll just use rank 12 per block = 60 total, same as schoolbook. That won't improve.
    #
    # Let me use the known 5*11 = 55 approach. For 2x3*3x2, rank 11:
    #
    # Actually, let me just code the Strassen-based approach for 2x2 blocks.
    # Split the 3 rows of B into pairs: rows (0,1) and row 2.
    # A * B = A * [B_top; B_bottom] where B_top is 2x10, B_bottom is 1x10.
    # = A_top * B_top + A_bottom * B_bottom
    # where A_top = A[:, :2] (2x2), A_bottom = A[:, 2:] (2x1).
    #
    # A_top * B_top: 2x2 * 2x10. Split B_top into 5 blocks of 2x2.
    # Each 2x2 * 2x2 = Strassen, rank 7. So 5*7 = 35 for this part.
    #
    # A_bottom * B_bottom: 2x1 * 1x10. 
    # A_bottom = [c, f]^T, B_bottom = [k, l, m, n, o, p, q, r, s, t] (1x10).
    # Result: [c*k+f*... wait, A_bottom is 2x1, B_bottom is 1x10.
    # A_bottom * B_bottom = [[c], [f]] * [k, l, ..., t] = [[c*k, c*l, ..., c*t],
    #                                                        [f*k, f*l, ..., f*t]]
    # This is 2*10 = 20 scalar products (rank 20).
    #
    # Total: 35 + 20 = 55!
    #
    # Let me verify: A = [[a, b, c], [d, e, f]]
    # A_top = [[a, b], [d, e]], A_bottom = [c, f]^T
    # B_top = B[0:2, :] (2x10), B_bottom = B[2, :] (1x10)
    # A*B = A_top * B_top + A_bottom * B_bottom
    # A_top * B_top: split into 5 blocks of 2x2, use Strassen (rank 7 each) = 35
    # A_bottom * B_bottom: 2x1 * 1x10, rank 20 (schoolbook)
    # Total rank: 35 + 20 = 55
    #
    # Now let's build the factor matrices.
    
    rank = 55
    U = [[0] * rank for _ in range(6)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(30)]  # 3*10 = 30 rows
    W = [[0] * rank for _ in range(20)]  # 10*2 = 20 rows
    
    col = 0
    # Part 1: A_top * B_top, 5 Strassen blocks
    for r in range(5):
        # Block r: columns 2r, 2r+1 of B_top
        # B_top block = [[e, f], [g, h]] where e=B[0][2r], f=B[0][2r+1],
        # g=B[1][2r], h=B[1][2r+1]
        # A_top = [[a, b], [d, e]] where a=A[0][0], b=A[0][1], d=A[1][0], e=A[1][1]
        # C_block = A_top * B_top_block (2x2)
        # P1 = a*f
        # P2 = (a+b)*h
        # P3 = a*(e-f)  [note: e here is B[0][2r], not A[1][1]]
        # Let me use different names.
        # A_top = [[a11, a12], [a21, a22]]
        # B_block = [[b11, b12], [b21, b22]]
        # P1 = a11*b12
        # P2 = (a11+a12)*b22
        # P3 = a11*(b11-b12)
        # P4 = a22*(b21-b22)
        # P5 = (a11+a22)*(b11+b22)
        # P6 = (a21-a22)*(b11+b12)
        # P7 = (a12-a21)*(b21+b22)
        # C11 = P5 + P4 - P2 + P6
        # C12 = P1 + P2 - P5 - P7
        # C21 = P3 + P5 - P1 - P4
        # C22 = P1 + P3 - P2 + P7
        
        # U rows: 0=a11=A[0][0], 1=a12=A[0][1], 2=a21=A[1][0], 3=a22=A[1][1]
        # V rows: for B_block, rows are 0,1 of B (since B_top = B[0:2,:])
        # V row for B[0][2r] = 0*10 + 2r = 2r
        # V row for B[0][2r+1] = 0*10 + 2r+1 = 2r+1
        # V row for B[1][2r] = 1*10 + 2r = 10+2r
        # V row for B[1][2r+1] = 1*10 + 2r+1 = 10+2r+1
        # W rows: C_block[i][j] -> W[(2r+j)*2 + i]
        # C[0][0] -> W[4r+0], C[0][1] -> W[4r+1], C[1][0] -> W[4r+2], C[1][1] -> W[4r+3]
        
        # P1 = a11 * b12: U[0]=1, V[2r+1]=1, W: C12 += 1, C22 += 1
        U[0][col] = 1
        V[2*r + 1][col] = 1
        W[4*r + 1][col] = 1  # C[0][1]
        W[4*r + 3][col] = 1  # C[1][1]
        col += 1
        
        # P2 = (a11+a12) * b22: U[0]=1, U[1]=1, V[10+2r+1]=1, W: C11 -= 1, C12 += 1, C22 -= 1
        U[0][col] = 1
        U[1][col] = 1
        V[10 + 2*r + 1][col] = 1
        W[4*r + 0][col] = -1  # C[0][0]
        W[4*r + 1][col] = 1   # C[0][1]
        W[4*r + 3][col] = -1  # C[1][1]
        col += 1
        
        # P3 = a11 * (b11-b12): U[0]=1, V[2r]=1, V[2r+1]=-1, W: C21 += 1, C22 += 1
        U[0][col] = 1
        V[2*r][col] = 1
        V[2*r + 1][col] = -1
        W[4*r + 2][col] = 1  # C[1][0]
        W[4*r + 3][col] = 1  # C[1][1]
        col += 1
        
        # P4 = a22 * (b21-b22): U[3]=1, V[10+2r]=1, V[10+2r+1]=-1, W: C11 += 1, C21 -= 1
        U[3][col] = 1
        V[10 + 2*r][col] = 1
        V[10 + 2*r + 1][col] = -1
        W[4*r + 0][col] = 1  # C[0][0]
        W[4*r + 2][col] = -1 # C[1][0]
        col += 1
        
        # P5 = (a11+a22) * (b11+b22): U[0]=1, U[3]=1, V[2r]=1, V[10+2r+1]=1,
        # W: C11 += 1, C12 -= 1, C21 += 1
        U[0][col] = 1
        U[3][col] = 1
        V[2*r][col] = 1
        V[10 + 2*r + 1][col] = 1
        W[4*r + 0][col] = 1  # C[0][0]
        W[4*r + 1][col] = -1 # C[0][1]
        W[4*r + 2][col] = 1  # C[1][0]
        col += 1
        
        # P6 = (a21-a22) * (b11+b12): U[2]=1, U[3]=-1, V[2r]=1, V[2r+1]=1,
        # W: C11 += 1
        U[2][col] = 1
        U[3][col] = -1
        V[2*r][col] = 1
        V[2*r + 1][col] = 1
        W[4*r + 0][col] = 1  # C[0][0]
        col += 1
        
        # P7 = (a12-a21) * (b21+b22): U[1]=1, U[2]=-1, V[10+2r]=1, V[10+2r+1]=1,
        # W: C12 -= 1, C22 += 1
        U[1][col] = 1
        U[2][col] = -1
        V[10 + 2*r][col] = 1
        V[10 + 2*r + 1][col] = 1
        W[4*r + 1][col] = -1 # C[0][1]
        W[4*r + 3][col] = 1  # C[1][1]
        col += 1
    
    # Part 2: A_bottom * B_bottom, 2x1 * 1x10, schoolbook (20 products)
    # A_bottom = [c, f]^T = A[0][2], A[1][2]
    # B_bottom = B[2, :] (1x10)
    # C[i][j] += A[i][2] * B[2][j]
    # W row for C[i][j] = j*2 + i
    for j in range(10):
        for i in range(2):
            U[i * 3 + 2][col] = 1  # A[i][2]
            V[20 + j][col] = 1     # B[2][j]
            W[j * 2 + i][col] = 1  # C[i][j]
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
