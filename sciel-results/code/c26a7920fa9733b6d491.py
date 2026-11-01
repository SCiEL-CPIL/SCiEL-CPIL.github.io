"""Strassen rank-7 exact decomposition for 2x2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm gives a rank-7 exact decomposition.
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)
    # W has shape (p*n, rank) = (4, 7)
    #
    # Indexing:
    #   U[i*m + j, c] for i in {0,1}, j in {0,1}
    #   V[j*p + k, c] for j in {0,1}, k in {0,1}
    #   W[k*n + i, c] for k in {0,1}, i in {0,1}  (official trace convention)
    #
    # Strassen's 7 products:
    #   p1 = (a+d)(e+h)
    #   p2 = (c+d)e
    #   p3 = a(f-h)
    #   p4 = d(g-e)
    #   p5 = (a+b)h
    #   p6 = (c-a)(e+f)
    #   p7 = (b-d)(g+h)
    #
    # Result:
    #   C[0,0] = p1 + p4 - p5 + p7
    #   C[0,1] = p3 + p5
    #   C[1,0] = p2 + p4
    #   C[1,1] = p1 - p2 + p3 + p6
    #
    # A = [[a, b], [c, d]], indices: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    #   U row index: i*m+j => U[0]=a, U[1]=b, U[2]=c, U[3]=d
    # B = [[e, f], [g, h]], indices: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    #   V row index: j*p+k => V[0]=e, V[1]=f, V[2]=g, V[3]=h
    # C = [[C00, C01], [C10, C11]]
    #   W row index: k*n+i => W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11

    rank = 7
    # Initialize matrices
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # p1 = (a+d)(e+h)
    # A-part: a + d => U[0,0]=1, U[3,0]=1
    # B-part: e + h => V[0,0]=1, V[3,0]=1
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    # C-part: C00 += p1, C11 += p1
    W[0][0] = 1
    W[3][0] = 1

    # p2 = (c+d)e
    # A-part: c + d => U[2,1]=1, U[3,1]=1
    # B-part: e => V[0,1]=1
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    # C-part: C10 += p2, C11 -= p2
    W[2][1] = 1
    W[3][1] = -1

    # p3 = a(f-h)
    # A-part: a => U[0,2]=1
    # B-part: f - h => V[1,2]=1, V[3,2]=-1
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    # C-part: C01 += p3, C11 += p3
    W[1][2] = 1
    W[3][2] = 1

    # p4 = d(g-e)
    # A-part: d => U[3,3]=1
    # B-part: g - e => V[2,3]=1, V[0,3]=-1
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    # C-part: C00 += p4, C10 += p4
    W[0][3] = 1
    W[2][3] = 1

    # p5 = (a+b)h
    # A-part: a + b => U[0,4]=1, U[1,4]=1
    # B-part: h => V[3,4]=1
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    # C-part: C00 -= p5, C01 += p5
    W[0][4] = -1
    W[1][4] = 1

    # p6 = (c-a)(e+f)
    # A-part: c - a => U[2,5]=1, U[0,5]=-1
    # B-part: e + f => V[0,5]=1, V[1,5]=1
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    # C-part: C11 += p6
    W[3][5] = 1

    # p7 = (b-d)(g+h)
    # A-part: b - d => U[1,6]=1, U[3,6]=-1
    # B-part: g + h => V[2,6]=1, V[3,6]=1
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    # C-part: C00 += p7
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
