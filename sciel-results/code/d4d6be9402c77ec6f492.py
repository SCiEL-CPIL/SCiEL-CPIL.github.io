import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    deadline = time.time() + 110  # safety margin before 120s limit
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for t in range(num_pairs):
        base = t * 7
        k0 = 2 * t      # first column in pair
        k1 = 2 * t + 1  # second column in pair
        
        # M1 = (A00 + A11) * (B00 + B11)
        c = base + 0
        U[0][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        V[p + k1][c] = 1
        W[k0][c] = 1
        W[p + k1][c] = 1
        
        # M2 = (A10 + A11) * B00
        c = base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        W[p + k0][c] = 1
        W[p + k1][c] = -1
        
        # M3 = A00 * (B01 - B11)
        c = base + 2
        U[0][c] = 1
        V[k1][c] = 1
        V[p + k1][c] = -1
        W[k1][c] = 1
        W[p + k1][c] = 1
        
        # M4 = A11 * (B10 - B00)
        c = base + 3
        U[3][c] = 1
        V[p + k0][c] = 1
        V[k0][c] = -1
        W[k0][c] = 1
        W[p + k0][c] = 1
        
        # M5 = (A00 + A01) * B11
        c = base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[p + k1][c] = 1
        W[k0][c] = -1
        W[k1][c] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[k0][c] = 1
        V[k1][c] = 1
        W[p + k1][c] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[p + k0][c] = 1
        V[p + k1][c] = 1
        W[k0][c] = 1
        
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
