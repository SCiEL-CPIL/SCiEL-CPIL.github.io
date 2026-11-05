"""Tensor decomposition for 3x4x5 matrix multiplication using Strassen blocks."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_decomposition():
    """Build a low-rank decomposition using Strassen 2x2x2 blocks with explicit (i,j,k) tracking."""
    n, m, p = 3, 4, 5
    # Each term is (U_entries, V_entries, W_entries) where each is a dict: key->value
    # U key: (i, j) for A[i,j], V key: (j, k) for B[j,k], W key: (k, i) for C[k,i]
    terms = []  # list of (u_dict, v_dict, w_dict)

    def add_term(u, v, w):
        terms.append((u, v, w))

    # Strassen 2x2x2: A is 2x2 (rows a0,a1; cols j0,j1), B is 2x2 (rows j0,j1; cols k0,k1), C is 2x2 (rows c0,c1; cols k0,k1)
    # We compute C = A*B where A[i][j], B[j][k], C[i][k]
    # Strassen:
    # M1=(A[a0,j0]+A[a1,j0])*(B[j0,k0]-B[j0,k1]) + (A[a0,j1]+A[a1,j1])*(B[j1,k0]-B[j1,k1])
    #   = (row a0 + row a1) of A * (col k0 - col k1) of B, contributes to C[c1][k1]... 
    # Actually, let me use the standard formulation where we combine over j:
    # For 2x2x2: A(2x2)*B(2x2)->C(2x2)
    # M1 = (A00+A10)(B00-B10) + (A01+A11)(B01-B11)  -- this is (row0+row1 of A) dot (col0-col1 of B)
    #   contributes to C11
    # M2 = (A00+A10)*B10 + (A01+A11)*B11  -- (row0+row1) dot col1
    #   contributes to C01 and C11
    # etc.
    # 
    # The rank-7 decomposition of 2x2x2 matrix multiplication:
    # Each "multiplier" M_r is a bilinear form in A and B.
    # M1 = (A00+A10)(B00-B10) + (A01+A11)(B01-B11)
    # M2 = (A00+A10)(B00+B10) + (A01+A11)(B01+B11)  
    # M3 = A00*(B00-B10) + A01*(B01-B11)
    # M4 = A10*B00 + A11*B01
    # M5 = A00*B00 + A01*B01
    # M6 = (A10-A00)*B00 + (A11-A01)*B01
    # M7 = (A10-A00)*(B01-B00) + (A11-A01)*(B11-B10)
    #
    # C00 = M5 + M4 - M2 + M6
    # C01 = M3 + M1 - M4 - M7  ... let me look up the correct formula.
    #
    # Standard Strassen for 2x2:
    # M1 = (a00+a10)(b00-b10) + (a01+a11)(b01-b11)
    # M2 = (a00+a10)*b10 + (a01+a11)*b11
    # M3 = a00*(b00-b10) + a01*(b01-b11)
    # M4 = a10*b00 + a11*b01
    # M5 = a00*b00 + a01*b01
    # M6 = (a10-a00)*b00 + (a11-a01)*b01
    # M7 = (a10-a00)*(b01-b00) + (a11-a01)*(b11-b10)
    #
    # c00 = m5 + m4 - m2 + m6
    # c01 = m3 + m1 - m4 - m7
    # c10 = m2 + m7
    # c11 = m1 + m5 - m6 - m3
    #
    # Wait, I need to be careful with indices. Let me use:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    #
    # Strassen multipliers (using 0-indexed: A[i][j], B[j][k], C[i][k]):
    # M1 = (A[0][0]+A[1][0])*(B[0][0]-B[0][1]) + (A[0][1]+A[1][1])*(B[1][0]-B[1][1])
    # M2 = (A[0][0]+A[1][0])*B[1][0] + (A[0][1]+A[1][1])*B[1][1]
    # M3 = A[0][0]*(B[0][0]-B[0][1]) + A[0][1]*(B[1][0]-B[1][1])
    # M4 = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    # M5 = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    # M6 = (A[1][0]-A[0][0])*B[0][0] + (A[1][1]-A[0][1])*B[1][0]
    # M7 = (A[1][0]-A[0][0])*(B[0][1]-B[0][0]) + (A[1][1]-A[0][1])*(B[1][1]-B[1][0])
    #
    # c00 = M5 + M4 - M2 + M6
    # c01 = M3 + M1 - M4 - M7
    # c10 = M2 + M7
    # c11 = M1 + M5 - M6 - M3

    def strassen222(a_i0, a_i1, b_j0, b_j1, b_k0, b_k1, c_i0, c_i1):
        """
        A rows: a_i0, a_i1 (indices in 0..n-1)
        A cols = B rows: b_j0, b_j1 (indices in 0..m-1)  
        B cols = C cols: b_k0, b_k1 (indices in 0..p-1)
        C rows: c_i0, c_i1 (indices in 0..n-1)
        
        This computes the 2x2x2 block: A[a_i0:a_i1+1, b_j0:b_j1+1] * B[b_j0:b_j1+1, b_k0:b_k1+1] -> C[c_i0:c_i1+1, b_k0:b_k1+1]
        """
        # M1 = (A[a_i0][b_j0]+A[a_i1][b_j0])*(B[b_j0][b_k0]-B[b_j0][b_k1]) + (A[a_i0][b_j1]+A[a_i1][b_j1])*(B[b_j1][b_k0]-B[b_j1][b_k1])
        u1 = {(a_i0,b_j0):1, (a_i1,b_j0):1, (a_i0,b_j1):1, (a_i1,b_j1):1}
        v1 = {(b_j0,b_k0):1, (b_j0,b_k1):-1, (b_j1,b_k0):1, (b_j1,b_k1):-1}
        w1 = {(b_k1,c_i1):1}
        add_term(u1, v1, w1)
        
        # M2 = (A[a_i0][b_j0]+A[a_i1][b_j0])*B[b_j1][b_k0] + (A[a_i0][b_j1]+A[a_i1][b_j1])*B[b_j1][b_k1]
        u2 = {(a_i0,b_j0):1, (a_i1,b_j0):1, (a_i0,b_j1):1, (a_i1,b_j1):1}
        v2 = {(b_j1,b_k0):1, (b_j1,b_k1):1}
        w2 = {(b_k0,c_i0):-1, (b_k1,c_i1):1}
        add_term(u2, v2, w2)
        
        # M3 = A[a_i0][b_j0]*(B[b_j0][b_k0]-B[b_j0][b_k1]) + A[a_i0][b_j1]*(B[b_j1][b_k0]-B[b_j1][b_k1])
        u3 = {(a_i0,b_j0):1, (a_i0,b_j1):1}
        v3 = {(b_j0,b_k0):1, (b_j0,b_k1):-1, (b_j1,b_k0):1, (b_j1,b_k1):-1}
        w3 = {(b_k1,c_i1):-1, (b_k0,c_i0):1}
        add_term(u3, v3, w3)
        
        # M4 = A[a_i1][b_j0]*B[b_j0][b_k0] + A[a_i1][b_j1]*B[b_j1][b_k0]
        u4 = {(a_i1,b_j0):1, (a_i1,b_j1):1}
        v4 = {(b_j0,b_k0):1, (b_j1,b_k0):1}
        w4 = {(b_k0,c_i0):1, (b_k1,c_i1):-1}
        add_term(u4, v4, w4)
        
        # M5 = A[a_i0][b_j0]*B[b_j0][b_k0] + A[a_i0][b_j1]*B[b_j1][b_k0]
        u5 = {(a_i0,b_j0):1, (a_i0,b_j1):1}
        v5 = {(b_j0,b_k0):1, (b_j1,b_k0):1}
        w5 = {(b_k0,c_i0):1}
        add_term(u5, v5, w5)
        
        # M6 = (A[a_i1][b_j0]-A[a_i0][b_j0])*B[b_j0][b_k0] + (A[a_i1][b_j1]-A[a_i0][b_j1])*B[b_j1][b_k0]
        u6 = {(a_i1,b_j0):1, (a_i0,b_j0):-1, (a_i1,b_j1):1, (a_i0,b_j1):-1}
        v6 = {(b_j0,b_k0):1, (b_j1,b_k0):1}
        w6 = {(b_k0,c_i0):1}
        add_term(u6, v6, w6)
        
        # M7 = (A[a_i1][b_j0]-A[a_i0][b_j0])*(B[b_j0][b_k1]-B[b_j0][b_k0]) + (A[a_i1][b_j1]-A[a_i0][b_j1])*(B[b_j1][b_k1]-B[b_j1][b_k0])
        u7 = {(a_i1,b_j0):1, (a_i0,b_j0):-1, (a_i1,b_j1):1, (a_i0,b_j1):-1}
        v7 = {(b_j0,b_k1):1, (b_j0,b_k0):-1, (b_j1,b_k1):1, (b_j1,b_k0):-1}
        w7 = {(b_k1,c_i1):-1}
        add_term(u7, v7, w7)

    # For rectangular blocks, we use elementary products
    def elementary21(a_i, b_j, b_k, c_i):
        """A[a_i][b_j] * B[b_j][b_k] -> C[c_i][b_k]"""
        add_term({(a_i,b_j):1}, {(b_j,b_k):1}, {(b_k,c_i):1})

    def elementary22(a_i, b_j, b_k, c_i):
        """Same as elementary21 (naming for 2x2 context)"""
        elementary21(a_i, b_j, b_k, c_i)

    # Now assemble the 3x4x5 multiplication.
    # A is 3x4 (rows 0-2, cols 0-3), B is 4x5 (rows 0-3, cols 0-4), C is 3x5 (rows 0-2, cols 0-4)
    # 
    # Split A rows: {0,1} and {2}
    # Split A cols / B rows: {0,1} and {2,3}
    # Split B cols / C cols: {0,1} and {2,3,4}
    # Split C rows: {0,1} and {2}
    #
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]  (2x2x2 + 2x2x2)
    # C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:4]*B[2:4, 2:5]  (2x2x3 + 2x2x3)
    # C[2:3, 0:2] = A[2:3, 0:2]*B[0:2, 0:2] + A[2:3, 2:4]*B[2:4, 0:2]  (1x2x2 + 1x2x2)
    # C[2:3, 2:5] = A[2:3, 0:2]*B[0:2, 2:5] + A[2:3, 2:4]*B[2:4, 2:5]  (1x2x3 + 1x2x3)

    # C11: 2x2x2 + 2x2x2
    strassen222(0, 1, 0, 1, 0, 1, 0, 1)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen222(0, 1, 2, 3, 0, 1, 0, 1)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]

    # C12: 2x2x3 + 2x2x3 (use Strassen for first 2 cols, elementary for 3rd)
    # A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]
    # Split B[0:2,2:5] into B[0:2,2:4] (2 cols) and B[0:2,4:5] (1 col)
    strassen222(0, 1, 0, 1, 2, 3, 0, 1)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # Elementary for B[0:2,4:5]: A[0:2,0:2] * B[0:2,4] -> C[0:2,4]
    elementary21(0, 0, 4, 0)  # A[0][0]*B[0][4] -> C[0][4]
    elementary21(0, 1, 4, 0)  # A[0][1]*B[1][4] -> C[0][4]
    elementary21(1, 0, 4, 1)  # A[1][0]*B[0][4] -> C[1][4]
    elementary21(1, 1, 4, 1)  # A[1][1]*B[1][4] -> C[1][4]
    
    # A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5]
    strassen222(0, 1, 2, 3, 2, 3, 0, 1)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    # Elementary for B[2:4,4:5]: A[0:2,2:4] * B[2:4,4] -> C[0:2,4]
    elementary21(0, 2, 4, 0)  # A[0][2]*B[2][4] -> C[0][4]
    elementary21(0, 3, 4, 0)  # A[0][3]*B[3][4] -> C[0][4]
    elementary21(1, 2, 4, 1)  # A[1][2]*B[2][4] -> C[1][4]
    elementary21(1, 3, 4, 1)  # A[1][3]*B[3][4] -> C[1][4]

    # C21: 1x2x2 + 1x2x2 (elementary)
    # A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2]
    elementary21(2, 0, 0, 2)  # A[2][0]*B[0][0] -> C[2][0]
    elementary21(2, 0, 1, 2)  # A[2][0]*B[0][1] -> C[2][1]
    elementary21(2, 1, 0, 2)  # A[2][1]*B[1][0] -> C[2][0]
    elementary21(2, 1, 1, 2)  # A[2][1]*B[1][1] -> C[2][1]
    # A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2]
    elementary21(2, 2, 0, 2)  # A[2][2]*B[2][0] -> C[2][0]
    elementary21(2, 2, 1, 2)  # A[2][2]*B[2][1] -> C[2][1]
    elementary21(2, 3, 0, 2)  # A[2][3]*B[3][0] -> C[2][0]
    elementary21(2, 3, 1, 2)  # A[2][3]*B[3][1] -> C[2][1]

    # C22: 1x2x3 + 1x2x3 (elementary)
    # A[2:3,0:2] * B[0:2,2:5] -> C[2:3,2:5]
    elementary21(2, 0, 2, 2)  # A[2][0]*B[0][2] -> C[2][2]
    elementary21(2, 0, 3, 2)  # A[2][0]*B[0][3] -> C[2][3]
    elementary21(2, 0, 4, 2)  # A[2][0]*B[0][4] -> C[2][4]
    elementary21(2, 1, 2, 2)  # A[2][1]*B[1][2] -> C[2][2]
    elementary21(2, 1, 3, 2)  # A[2][1]*B[1][3] -> C[2][3]
    elementary21(2, 1, 4, 2)  # A[2][1]*B[1][4] -> C[2][4]
    # A[2:3,2:4] * B[2:4,2:5] -> C[2:3,2:5]
    elementary21(2, 2, 2, 2)  # A[2][2]*B[2][2] -> C[2][2]
    elementary21(2, 2, 3, 2)  # A[2][2]*B[2][3] -> C[2][3]
    elementary21(2, 2, 4, 2)  # A[2][2]*B[2][4] -> C[2][4]
    elementary21(2, 3, 2, 2)  # A[2][3]*B[3][2] -> C[2][2]
    elementary21(2, 3, 3, 2)  # A[2][3]*B[3][3] -> C[2][3]
    elementary21(2, 3, 4, 2)  # A[2][3]*B[3][4] -> C[2][4]

    # Build U, V, W matrices
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i*m + j][r] = val
        for (j, k), val in v.items():
            V[j*p + k][r] = val
        for (k, i), val in w.items():
            W[k*n + i][r] = val

    return U, V, W, rank

def _build_schoolbook():
    """Fallback: elementary rank-60 decomposition."""
    n, m, p = 3, 4, 5
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    deadline = time.time() + 140.0
    U, V, W, rank = _build_decomposition()
    
    # Self-check with multiple random matrices
    import random
    def check():
        for seed in range(5):
            rng = random.Random(seed)
            n, m, p = 3, 4, 5
            A = [[rng.randint(-5,5) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-5,5) for _ in range(p)] for _ in range(m)]
            C = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j]*B[j][k]
                    C[i][k] = s
            Cd = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    Cd[i][k] = s
            if not all(Cd[i][k] == C[i][k] for i in range(n) for k in range(p)):
                return False
        return True
    
    if not check():
        # Fallback: schoolbook rank 60
        n, m, p = 3, 4, 5
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
