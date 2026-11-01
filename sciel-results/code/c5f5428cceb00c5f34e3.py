"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping:
    # U[i*m + j] corresponds to A[i,j]
    # V[j*p + k] corresponds to B[j,k]
    # W[k*n + i] corresponds to C[i,k]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B = [[c00, c01], [c10, c11]]
    
    # m1 = (a00 + a11)(b00 + b11)
    # U: a00 (0), a11 (3)
    # V: b00 (0), b11 (3)
    # W: c00 (0), c11 (3)
    U[0*2+0][0] = 1
    U[1*2+1][0] = 1
    V[0*2+0][0] = 1
    V[1*2+1][0] = 1
    W[0*2+0][0] = 1
    W[1*2+1][0] = 1
    
    # m2 = (a10 + a11)b00
    # U: a10 (2), a11 (3)
    # V: b00 (0)
    # W: c10 (1), c11 (3) with sign -1 for c11
    U[1*2+0][1] = 1
    U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1
    W[1*2+1][1] = -1
    
    # m3 = a00(b01 - b11)
    # U: a00 (0)
    # V: b01 (1), b11 (3) with sign -1
    # W: c01 (2), c11 (3)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1
    V[1*2+1][2] = -1
    W[1*2+0][2] = 1
    W[1*2+1][2] = 1
    
    # m4 = a11(b10 - b00)
    # U: a11 (3)
    # V: b10 (2), b00 (0) with sign -1
    # W: c00 (0), c10 (1)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1
    V[0*2+0][3] = -1
    W[0*2+0][3] = 1
    W[0*2+1][3] = 1
    
    # m5 = (a00 + a01)b11
    # U: a00 (0), a01 (1)
    # V: b11 (3)
    # W: c00 (0) with sign -1, c01 (2)
    U[0*2+0][4] = 1
    U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1
    W[1*2+0][4] = 1
    
    # m6 = (a10 - a00)(b00 + b01)
    # U: a10 (2), a00 (0) with sign -1
    # V: b00 (0), b01 (1)
    # W: c11 (3)
    U[1*2+0][5] = 1
    U[0*2+0][5] = -1
    V[0*2+0][5] = 1
    V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # m7 = (a01 - a11)(b10 + b11)
    # U: a01 (1), a11 (3) with sign -1
    # V: b10 (2), b11 (3)
    # W: c00 (0)
    U[0*2+1][6] = 1
    U[1*2+1][6] = -1
    V[1*2+0][6] = 1
    V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    if time.time() > deadline:
        # Fallback: The construction is static and O(1), so this should never trigger.
        # But if it did, we return the partially constructed (but likely complete) matrices.
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
