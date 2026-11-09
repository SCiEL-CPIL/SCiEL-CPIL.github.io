import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=3, p=3
    # U shape: (6, 17)
    # V shape: (9, 17)
    # W shape: (6, 17)
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1
        
    # Block 1: 2x2 by 2x2 (Strassen)
    # A indices: 0(a), 1(b), 3(c), 4(d)
    # B indices: 0(e), 1(f), 3(g), 4(h)
    # C indices: 0(p), 2(q), 1(r), 3(s)
    
    # M1 = (a+d)(e+h)
    add_term([0, 4], [0, 4], [0, 3]) # p=1, s=1
    # M2 = (c+d)e
    add_term([3, 4], [0], [1]) # r=1
    # M3 = a(f-h)
    add_term([0], [1, 4], [2]) # q=1
    # Wait, M3 contributes to q and s?
    # q = M3 + M5
    # s = M1 - M2 + M3 + M6
    # So M3 contributes +1 to q and +1 to s.
    # My add_term only takes +1. I need to handle signs or just use +1 and adjust W later?
    # The interface expects exact decomposition.
    # I can use negative values in U, V, W?
    # "integer/half-integer entries". Yes, negative integers are allowed.
    
    # Let's restart the Strassen part with signs.
    
    # Reset U, V, W to 0
    for r in range(rank):
        for i in range(n*m): U[i][r] = 0
        for i in range(m*p): V[i][r] = 0
        for i in range(p*n): W[i][r] = 0
    col = 0
    
    # M1 = (a+d)(e+h)
    # U: a(0)=1, d(4)=1
    # V: e(0)=1, h(4)=1
    # W: p(0)=1, s(3)=1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (c+d)e
    # U: c(3)=1, d(4)=1
    # V: e(0)=1
    # W: r(1)=1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M3 = a(f-h)
    # U: a(0)=1
    # V: f(1)=1, h(4)=-1
    # W: q(2)=1, s(3)=1
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = d(g-e)
    # U: d(4)=1
    # V: g(3)=1, e(0)=-1
    # W: p(0)=1, r(1)=1
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (a+b)h
    # U: a(0)=1, b(1)=1
    # V: h(4)=1
    # W: p(0)=-1, q(2)=1
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6 = (c-a)(e+f)
    # U: c(3)=1, a(0)=-1
    # V: e(0)=1, f(1)=1
    # W: s(3)=1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (b-d)(g+h)
    # U: b(1)=1, d(4)=-1
    # V: g(3)=1, h(4)=1
    # W: p(0)=1
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: 2x1 by 1x2
    # A: 2(a1), 5(a2)
    # B: 6(b1), 7(b2)
    # C: 0(c11), 2(c12), 1(c21), 3(c22)
    
    # a1*b1 -> C[0]
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    # a1*b2 -> C[2]
    U[2][col] = 1; V[7][col] = 1; W[2][col] = 1; col += 1
    # a2*b1 -> C[1]
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    # a2*b2 -> C[3]
    U[5][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    
    # Block 3: 2x2 by 2x1
    # A: 0, 1, 3, 4
    # B: 2(b1), 5(b2)
    # C: 4(c1), 5(c2)
    
    # A[0]*B[2] -> C[4]
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    # A[1]*B[5] -> C[4]
    U[1][col] = 1; V[5][col] = 1; W[4][col] = 1; col += 1
    # A[3]*B[2] -> C[5]
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    # A[4]*B[5] -> C[5]
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # Block 4: 2x1 by 1x1
    # A: 2, 5
    # B: 8
    # C: 4, 5
    
    # A[2]*B[8] -> C[4]
    U[2][col] = 1; V[8][col] = 1; W[4][col] = 1; col += 1
    # A[5]*B[8] -> C[5]
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
