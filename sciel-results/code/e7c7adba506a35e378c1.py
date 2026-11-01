"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    num_blocks = p // 2  # 5 blocks of 3x2
    rank = num_blocks * 11  # 55
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (20, 55)
    
    col = 0
    for block in range(num_blocks):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[p+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[1,k0]-B[0,k0])
        U[0][col] = 1
        V[p+k0][col] = 1; V[k0][col] = -1
        W[2*k1][col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[0,k1]-B[1,k1])
        U[4][col] = 1
        V[p+k1][col] = -1; V[k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1; U[1][col] = 1
        V[p+k1][col] = 1
        W[2*k0][col] = 1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1; U[4][col] = -1
        V[p+k0][col] = 1; V[p+k1][col] = 1
        W[2*k0][col] = -1; W[2*k0+1][col] = 1
        col += 1
        
        # S1 = A[0,2] * B[2,k0]
        U[2][col] = 1
        V[2*p+k0][col] = 1
        W[2*k0][col] = 1
        col += 1
        
        # S2 = A[0,2] * B[2,k1]
        U[2][col] = 1
        V[2*p+k1][col] = 1
        W[2*k1][col] = 1
        col += 1
        
        # S3 = A[1,2] * B[2,k0]
        U[5][col] = 1
        V[2*p+k0][col] = 1
        W[2*k0+1][col] = 1
        col += 1
        
        # S4 = A[1,2] * B[2,k1]
        U[5][col] = 1
        V[2*p+k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
