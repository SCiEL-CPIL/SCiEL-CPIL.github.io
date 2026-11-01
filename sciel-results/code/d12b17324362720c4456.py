"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 Strassen blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(8):
        if time.time() > deadline - 10:
            break
            
        # M1 = (a+d)(e+h)
        U[0][col] = 1
        U[3][col] = 1
        V[0*16 + 2*l][col] = 1
        V[1*16 + 2*l+1][col] = 1
        W[2*l*2 + 0][col] = 1
        W[2*l*2 + 3][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[2][col] = 1
        U[3][col] = 1
        V[0*16 + 2*l][col] = 1
        W[2*l*2 + 2][col] = 1
        W[2*l*2 + 3][col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[0][col] = 1
        V[0*16 + 2*l+1][col] = 1
        V[1*16 + 2*l+1][col] = -1
        W[2*l*2 + 1][col] = 1
        W[2*l*2 + 3][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[3][col] = 1
        V[1*16 + 2*l][col] = 1
        V[0*16 + 2*l][col] = -1
        W[2*l*2 + 0][col] = 1
        W[2*l*2 + 2][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[0][col] = 1
        U[1][col] = 1
        V[1*16 + 2*l+1][col] = 1
        W[2*l*2 + 0][col] = -1
        W[2*l*2 + 1][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[2][col] = 1
        U[0][col] = -1
        V[0*16 + 2*l][col] = 1
        V[0*16 + 2*l+1][col] = 1
        W[2*l*2 + 3][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[1][col] = 1
        U[3][col] = -1
        V[1*16 + 2*l][col] = 1
        V[1*16 + 2*l+1][col] = 1
        W[2*l*2 + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
