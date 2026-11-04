"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-46 decomposition for 3x3 * 3x8 matrix multiplication.
    
    Structure:
    - A[0:2,0:2] * B[0:2,0:8] (2x2 * 2x8): two Strassen 2x2*2x4 blocks = 14 mults
    - A[0:2,2] * B[2,0:8] (2x1 * 1x8): 16 mults
    - A[2,0:2] * B[0:2,0:8] (1x2 * 2x8): 16 mults
    - A[2,2] * B[2,0:8] (1x1 * 1x8): 8 mults
    Total: 46
    
    Index conventions:
    U[i*m+j, c]: A[i,j] coefficient in factor c
    V[j*p+k, c]: B[j,k] coefficient in factor c  
    W[k*n+i, c]: C[i,k] coefficient in factor c (trace convention)
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]   # 9 x 46
    V = [[0] * rank for _ in range(m * p)]   # 24 x 46
    W = [[0] * rank for _ in range(p * n)]   # 24 x 46
    
    def w_idx(i, k):
        """W index for C[i,k] = W[k*n + i]"""
        return k * n + i
    
    def u_idx(i, j):
        """U index for A[i,j] = i*m + j"""
        return i * m + j
    
    def v_idx(j, k):
        """V index for B[j,k] = j*p + k"""
        return j * p + k
    
    col = 0
    
    # === BLOCK 1: A[0:2,0:2] * B[0:2,0:4] → C[0:2,0:4] (Strassen 2x2*2x4, rank 7) ===
    # A sub: a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[3], d=A[1,1]=U[4]
    # B sub: e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[0,2]=V[2], h=B[0,3]=V[3]
    #         i=B[1,0]=V[8], j=B[1,1]=V[9], k=B[1,2]=V[10], l=B[1,3]=V[11]
    # C sub: C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[0,3]=W[3]
    #         C[1,0]=W[4], C[1,1]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    
    # m1 = (a+d)*(e+i)
    U[u_idx(0,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1; V[v_idx(1,0)][col] = 1
    W[w_idx(0,0)][col] = 1; W[w_idx(1,1)][col] = 1
    col += 1
    
    # m2 = (c+d)*e
    U[u_idx(1,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    W[w_idx(0,0)][col] = -1; W[w_idx(0,2)][col] = 1; W[w_idx(1,0)][col] = 1; W[w_idx(1,2)][col] = -1
    col += 1
    
    # m3 = a*(f-i)
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1; V[v_idx(1,1)][col] = -1
    W[w_idx(0,1)][col] = 1; W[w_idx(1,1)][col] = 1; W[w_idx(1,3)][col] = 1
    col += 1
    
    # m4 = d*(h-g)
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,3)][col] = 1; V[v_idx(0,2)][col] = -1
    W[w_idx(0,0)][col] = -1; W[w_idx(0,3)][col] = 1; W[w_idx(1,1)][col] = 1
    col += 1
    
    # m5 = (a+b)*l
    U[u_idx(0,0)][col] = 1; U[u_idx(0,1)][col] = 1
    V[v_idx(1,3)][col] = 1
    W[w_idx(0,1)][col] = -1; W[w_idx(0,2)][col] = 1; W[w_idx(1,0)][col] = -1; W[w_idx(1,2)][col] = -1
    col += 1
    
    # m6 = (c-a)*(e+i)
    U[u_idx(1,0)][col] = 1; U[u_idx(0,0)][col] = -1
    V[v_idx(0,0)][col] = 1; V[v_idx(1,0)][col] = 1
    W[w_idx(0,0)][col] = 1; W[w_idx(1,1)][col] = -1
    col += 1
    
    # m7 = (d-b)*(g+h)
    U[u_idx(1,1)][col] = 1; U[u_idx(0,1)][col] = -1
    V[v_idx(0,2)][col] = 1; V[v_idx(0,3)][col] = 1
    W[w_idx(0,1)][col] = -1; W[w_idx(0,2)][col] = 1; W[w_idx(1,0)][col] = -1; W[w_idx(1,2)][col] = 1
    col += 1
    
    # === BLOCK 2: A[0:2,0:2] * B[0:2,4:8] → C[0:2,4:8] (Strassen 2x2*2x4, rank 7) ===
    # B sub: e=B[0,4]=V[4], f=B[0,5]=V[5], g=B[0,6]=V[6], h=B[0,7]=V[7]
    #         i=B[1,4]=V[12], j=B[1,5]=V[13], k=B[1,6]=V[14], l=B[1,7]=V[15]
    # C sub: C[0,4]=W[12], C[0,5]=W[13], C[0,6]=W[14], C[0,7]=W[15]
    #         C[1,4]=W[16], C[1,5]=W[17], C[1,6]=W[18], C[1,7]=W[19]
    
    # m1 = (a+d)*(e+i)
    U[u_idx(0,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,4)][col] = 1; V[v_idx(1,4)][col] = 1
    W[w_idx(0,4)][col] = 1; W[w_idx(1,5)][col] = 1
    col += 1
    
    # m2 = (c+d)*e
    U[u_idx(1,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(0,4)][col] = -1; W[w_idx(0,6)][col] = 1; W[w_idx(1,4)][col] = 1; W[w_idx(1,6)][col] = -1
    col += 1
    
    # m3 = a*(f-i)
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,5)][col] = 1; V[v_idx(1,5)][col] = -1
    W[w_idx(0,5)][col] = 1; W[w_idx(1,5)][col] = 1; W[w_idx(1,7)][col] = 1
    col += 1
    
    # m4 = d*(h-g)
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,7)][col] = 1; V[v_idx(0,6)][col] = -1
    W[w_idx(0,4)][col] = -1; W[w_idx(0,7)][col] = 1; W[w_idx(1,5)][col] = 1
    col += 1
    
    # m5 = (a+b)*l
    U[u_idx(0,0)][col] = 1; U[u_idx(0,1)][col] = 1
    V[v_idx(1,7)][col] = 1
    W[w_idx(0,5)][col] = -1; W[w_idx(0,6)][col] = 1; W[w_idx(1,4)][col] = -1; W[w_idx(1,6)][col] = -1
    col += 1
    
    # m6 = (c-a)*(e+i)
    U[u_idx(1,0)][col] = 1; U[u_idx(0,0)][col] = -1
    V[v_idx(0,4)][col] = 1; V[v_idx(1,4)][col] = 1
    W[w_idx(0,4)][col] = 1; W[w_idx(1,5)][col] = -1
    col += 1
    
    # m7 = (d-b)*(g+h)
    U[u_idx(1,1)][col] = 1; U[u_idx(0,1)][col] = -1
    V[v_idx(0,6)][col] = 1; V[v_idx(0,7)][col] = 1
    W[w_idx(0,5)][col] = -1; W[w_idx(0,6)][col] = 1; W[w_idx(1,4)][col] = -1; W[w_idx(1,6)][col] = 1
    col += 1
    
    # === BLOCK 3: A[0:2,2] * B[2,0:8] → C[0:2,0:8] (16 mults) ===
    # A[0,2]=U[2], A[1,2]=U[5]
    # B[2,k]=V[16+k] for k=0..7
    # C[0,k]=W[k] for k=0..7
    # C[1,k]=W[8+k] for k=0..7
    for i in range(2):
        for k in range(8):
            U[u_idx(i, 2)][col] = 1
            V[v_idx(2, k)][col] = 1
            W[w_idx(i, k)][col] = 1
            col += 1
    
    # === BLOCK 4: A[2,0:2] * B[0:2,0:8] → C[2,0:8] (16 mults) ===
    # A[2,0]=U[6], A[2,1]=U[7]
    # B[0,k]=V[k], B[1,k]=V[8+k] for k=0..7
    # C[2,k]=W[16+k] for k=0..7
    for j in range(2):
        for k in range(8):
            U[u_idx(2, j)][col] = 1
            V[v_idx(j, k)][col] = 1
            W[w_idx(2, k)][col] = 1
            col += 1
    
    # === BLOCK 5: A[2,2] * B[2,0:8] → C[2,0:8] (8 mults) ===
    # A[2,2]=U[8]
    # B[2,k]=V[16+k] for k=0..7
    # C[2,k]=W[16+k] for k=0..7
    for k in range(8):
        U[u_idx(2, 2)][col] = 1
        V[v_idx(2, k)][col] = 1
        W[w_idx(2, k)][col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() - start_time > 140:
        return None
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
