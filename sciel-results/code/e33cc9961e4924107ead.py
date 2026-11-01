"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT:
        # Return a valid, though not necessarily optimal, decomposition immediately
        # Fallback to schoolbook rank 24 if we run out of time (though this is fast)
        return _schoolbook_decomposition()

    n, m, p = DIMENSIONS
    # Rank is 3 * 7 = 21
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen constants for 2x2 multiplication
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*6 + k
    # W indices: k*2 + i
    
    # We process 3 blocks of 2x2
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V row indices for this block
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # W row indices for this block
        w_c00 = k0 * n + 0 # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n + 0 # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]
        
        # Strassen steps 0..6
        # Step 0: P1 = (a+d)(e+h)
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        # Step 1: P2 = (c+d)e
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        # Step 2: P3 = a(f-h)
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        # Step 3: P4 = d(g-e)
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        # Step 4: P5 = (a+b)h
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        # Step 5: P6 = (c-a)(e+f)
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        # Step 6: P7 = (b-d)(g+h)
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1

    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
