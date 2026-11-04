"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build rank-54 Strassen decomposition
    rank_base = 54
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] -= 1
        col += 1
        # M2: A00(B01-B11)
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+k_base][col] += 1
        V[j_base*p+k_base][col] -= 1
        W[k_base*n+i_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        # M4: (A00+A01)B11
        U[i_base*m+j_base][col] += 1
        U[i_base*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base][col] += 1
        U[i_base*m+j_base][col] -= 1
        V[j_base*p+k_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1)][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] -= 1
        V[(j_base+1)*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    if col != rank_base:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][i*m*p + j*p + k] = 1
                    V[j*p+k][i*m*p + j*p + k] = 1
                    W[k*n+i][i*m*p + j*p + k] = 1
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    # Structural change: try to find rank-53 via local search
    # Try different k-pairings to see if a better decomposition emerges
    # or use a heuristic to compress the decomposition
    
    # Try: can we use a rank-11 decomposition for 2x3*3x2 blocks?
    # Partition j as {0,1,2} and {3,4}
    # For {3,4}: 3 Strassen blocks = 21
    # For {0,1,2}: 3 blocks of 2x3*3x2, need rank per block
    
    # Try a 2x3*3x2 decomposition with rank 10:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    
    # Attempt rank-10:
    # M0 = (A00+A10)(B00+B01) -> C[0,0]+=, C[0,1]+=, C[1,0]+=, C[1,1]+=
    # M1 = (A01+A11)(B10+B11) -> similar
    # M2 = (A02+A12)(B20+B21) -> similar
    # M3 = (A00-A10)B00
    # M4 = (A00-A10)B01
    # M5 = (A01-A11)B10
    # M6 = (A01-A11)B11
    # M7 = (A02-A12)B20
    # M8 = (A02-A12)B21
    # That's 9 terms. Let me check if this works.
    
    # Actually, let me just return rank 54 since finding a lower rank
    # requires significant new algebraic insight.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
