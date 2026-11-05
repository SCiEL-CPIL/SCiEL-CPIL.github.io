"""Strassen-based rank-59 decomposition for 2x3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Partition: m=3 into 2+1, p=10 into 2+8
    # Block A: 2x2x2 -> Strassen rank 7
    # Block B: 2x1x2 -> schoolbook rank 4
    # Block C: 2x3x8 -> schoolbook rank 48
    # Total rank = 7 + 4 + 48 = 59
    
    rank = 59
    U = [[0] * rank for _ in range(n * m)]   # 6 x 59
    V = [[0] * rank for _ in range(m * p)]   # 30 x 59
    W = [[0] * rank for _ in range(p * n)]   # 20 x 59
    
    col = 0
    
    # Block A: 2x2x2 Strassen, rank 7
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(g-h), p4 = d(f-g)
    # p5 = (a+b)h, p6 = (c-a)(e+g), p7 = (b-d)(f+h)
    # C00 = p1+p4-p5-p7, C01 = p3+p5, C10 = p2+p4, C11 = p1-p2+p3-p6
    
    strassen = [
        # (U_coeffs for A, V_coeffs for B, W_coeffs for C)
        # U: [a, b, c, d] -> [A00, A01, A10, A11]
        # V: [e, f, g, h] -> [B00, B01, B10, B11]
        # W: [C00, C01, C10, C11]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # p2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),  # p3
        ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),  # p4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5
        ([0,0,1,-1], [1,0,1,0], [0,0,0,-1]), # p6
        ([0,1,0,-1], [0,1,0,1], [-1,0,0,0]), # p7
    ]
    
    for t in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen[t]
        # U: i=0..1, j=0..1 -> index i*3+j
        for i in range(2):
            for j in range(2):
                U[i*3+j][col+t] = u_coeffs[i*2+j]
        # V: j=0..1, k=0..1 -> index j*10+k
        for j in range(2):
            for k in range(2):
                V[j*10+k][col+t] = v_coeffs[j*2+k]
        # W: k=0..1, i=0..1 -> index k*2+i
        for k in range(2):
            for i in range(2):
                W[k*2+i][col+t] = w_coeffs[k*2+i]
    col += 7
    
    # Block B: 2x1x2 schoolbook, rank 4 (i=0..1, j=2, k=0..1)
    for i in range(2):
        for k in range(2):
            U[i*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    # Block C: 2x3x8 schoolbook, rank 48 (i=0..1, j=0..2, k=2..9)
    for i in range(2):
        for j in range(3):
            for k in range(2, 10):
                U[i*3+j][col] = 1
                V[j*10+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
