"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for each 2x2 block of B, giving rank 4*7=28
    num_blocks = p // 2  # 4 blocks of 2x2
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen coefficients for 2x2 x 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    
    # For each block b (0..3), B block is B[:, 2b:2b+2]
    # A entries: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B block entries: (0,2b)=e, (0,2b+1)=f, (1,2b)=g, (1,2b+1)=h
    
    for b in range(num_blocks):
        base = b * 7
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1  # A[0,0]
        U[3][base + 0] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 0] += 1  # B[0,2b]
        V[1 * p + 2 * b + 1][base + 0] += 1  # B[1,2b+1]
        # W: C[0,2b] = M1 + M4 - M5 + M7, C[0,2b+1] = M3 + M5, C[1,2b] = M2 + M4, C[1,2b+1] = M1 - M2 + M3 + M6
        W[2 * b * n + 0][base + 0] += 1  # C[0,2b]
        W[(2 * b + 1) * n + 1][base + 0] += 1  # C[1,2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1  # A[1,0]
        U[3][base + 1] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 1] += 1  # B[0,2b]
        W[2 * b * n + 1][base + 1] += 1  # C[1,2b]
        W[(2 * b + 1) * n + 1][base + 1] -= 1  # C[1,2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1  # A[0,0]
        V[0 * p + 2 * b + 1][base + 2] += 1  # B[0,2b+1]
        V[1 * p + 2 * b + 1][base + 2] -= 1  # B[1,2b+1]
        W[(2 * b + 1) * n + 0][base + 2] += 1  # C[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 2] += 1  # C[1,2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1  # A[1,1]
        V[1 * p + 2 * b][base + 3] += 1  # B[1,2b]
        V[0 * p + 2 * b][base + 3] -= 1  # B[0,2b]
        W[2 * b * n + 0][base + 3] += 1  # C[0,2b]
        W[2 * b * n + 1][base + 3] += 1  # C[1,2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1  # A[0,0]
        U[1][base + 4] += 1  # A[0,1]
        V[1 * p + 2 * b + 1][base + 4] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 4] -= 1  # C[0,2b]
        W[(2 * b + 1) * n + 0][base + 4] += 1  # C[1,2b]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1  # A[1,0]
        U[0][base + 5] -= 1  # A[0,0]
        V[0 * p + 2 * b][base + 5] += 1  # B[0,2b]
        V[0 * p + 2 * b + 1][base + 5] += 1  # B[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 5] += 1  # C[1,2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1  # A[0,1]
        U[3][base + 6] -= 1  # A[1,1]
        V[1 * p + 2 * b][base + 6] += 1  # B[1,2b]
        V[1 * p + 2 * b + 1][base + 6] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 6] += 1  # C[0,2b]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
