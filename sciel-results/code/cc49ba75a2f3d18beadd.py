# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The decomposition is constructed programmatically to ensure correctness 
    of the Strassen formulas, but uses static lists for speed.
    
    Interface:
    U: (n*m, rank) -> (4, 7)
    V: (m*p, rank) -> (4, 7)
    W: (p*n, rank) -> (4, 7)
    
    Indexing conventions:
    U[i*m + j, c] corresponds to A[i, j]
    V[j*p + k, c] corresponds to B[j, k]
    W[k*n + i, c] corresponds to C[i, k]
    
    n=m=p=2.
    """
    # Deadline guard (trivial for static solution, but included per directive)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Rank 7 is the proven minimum for 2x2 matrix multiplication.
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's 7 products:
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    
    # Output reconstruction:
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5

    # --- Product 0: m0 = (a00 + a11)(b00 + b11) ---
    U[0][0] = 1   # a00
    U[3][0] = 1   # a11
    V[0][0] = 1   # b00
    V[3][0] = 1   # b11
    W[0][0] = 1   # c00 (k=0, i=0)
    W[3][0] = 1   # c11 (k=1, i=1)

    # --- Product 1: m1 = (a10 + a11)b00 ---
    U[2][1] = 1   # a10
    U[3][1] = 1   # a11
    V[0][1] = 1   # b00
    W[1][1] = 1   # c10 (k=0, i=1)
    W[3][1] = -1  # c11

    # --- Product 2: m2 = a00(b01 - b11) ---
    U[0][2] = 1   # a00
    V[1][2] = 1   # b01
    V[3][2] = -1  # b11
    W[2][2] = 1   # c01 (k=1, i=0)
    W[3][2] = 1   # c11

    # --- Product 3: m3 = a11(b10 - b00) ---
    U[3][3] = 1   # a11
    V[2][3] = 1   # b10
    V[0][3] = -1  # b00
    W[0][3] = 1   # c00
    W[1][3] = 1   # c10

    # --- Product 4: m4 = (a00 + a01)b11 ---
    U[0][4] = 1   # a00
    U[1][4] = 1   # a01
    V[3][4] = 1   # b11
    W[0][4] = -1  # c00
    W[2][4] = 1   # c01

    # --- Product 5: m5 = (a10 - a00)(b00 + b01) ---
    U[2][5] = 1   # a10
    U[0][5] = -1  # a00
    V[0][5] = 1   # b00
    V[1][5] = 1   # b01
    W[3][5] = 1   # c11

    # --- Product 6: m6 = (a01 - a11)(b10 + b11) ---
    U[1][6] = 1   # a01
    U[3][6] = -1  # a11
    V[2][6] = 1   # b10
    V[3][6] = 1   # b11
    W[0][6] = 1   # c00

    # Check deadline (should never trigger for static list construction)
    if time.time() > deadline:
         # Fallback: return what we have (which is complete)
         pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
