import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Rank calculation
    # M_222 rank 7
    # M_221 rank 4
    # M_224 = 2*M_222 -> 14
    # M_225 = M_223 + M_222 = (M_222 + M_221) + M_222 = 3*M_222 + M_221 -> 21 + 4 = 25? 
    # Wait.
    # M_223 = M_222 + M_221 -> 7 + 4 = 11.
    # M_225 = M_223 + M_222 -> 11 + 7 = 18.
    # Total 14 + 18 = 32.
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32
    
    col = 0
    
    # Helper to add a Strassen block for cols k0, k1
    def add_strassen(k0, k1):
        nonlocal col
        # 7 factors
        
        # M1: (a00+a11)(b00+b11)
        # U: 00, 11
        # V: 0k0, 1k1
        # W: C00, C11 (indices 2*k0+0, 2*k1+1)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M2: (a10+a11)b00
        # U: 10, 11
        # V: 0k0
        # W: C10, C11
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        
        # M3: a00(b01-b11)
        # U: 00
        # V: 0k1, 1k1
        # W: C01, C11
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M4: a11(b10-b00)
        # U: 11
        # V: 1k0, 0k0
        # W: C00, C10
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M5: (a00+a01)b11
        # U: 00, 01
        # V: 1k1
        # W: C00, C01
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        
        # M6: (a10-a00)(b00+b01)
        # U: 10, 00
        # V: 0k0, 0k1
        # W: C11
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7: (a01-a11)(b10+b11)
        # U: 01, 11
        # V: 1k0, 1k1
        # W: C00
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    # Helper to add a Trivial block for col k
    def add_trivial(k):
        nonlocal col
        # 4 factors
        # 1. a00 b00 -> C00
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 2. a01 b10 -> C00
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 3. a10 b00 -> C10
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        # 4. a11 b10 -> C10
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Block 1: Cols 0-3 (M_224)
    # Strassen 0,1
    add_strassen(0, 1)
    # Strassen 2,3
    add_strassen(2, 3)
    
    # Block 2: Cols 4-8 (M_225)
    # M_223: Strassen 4,5 + Trivial 6
    add_strassen(4, 5)
    add_trivial(6)
    # M_222: Strassen 7,8
    add_strassen(7, 8)
    
    if time.time() > deadline:
        # Return what we have, or raise? 
        # The task says "return the best result found".
        # But if we don't finish, the rank is wrong (matrices are full of zeros).
        # However, 32 iterations is very fast.
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
