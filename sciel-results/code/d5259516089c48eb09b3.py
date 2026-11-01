"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x2 by 2x5 multiplication into:
    # 1. 2x2 by 2x2 for columns 0,1 (Rank 7)
    # 2. 2x2 by 2x2 for columns 2,3 (Rank 7)
    # 3. 2x2 by 2x1 for column 4 (Rank 4)
    # Total Rank = 7 + 7 + 4 = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Helper to add a Strassen decomposition for 2x2 by 2x2
    # U, V, W for the block are added to the global U, V, W
    def add_strassen_222(start_rank, k_offset):
        # k_offset is the column index in B (0 or 2)
        # U indices: i*m+j (0..3)
        # V indices: j*p+k (j*5+k)
        # W indices: k*n+i (k*2+i)
        
        # M1: A00(B11-B10)
        U[0][start_rank + 0] = 1
        V[1*p + (k_offset+1)][start_rank + 0] = 1
        V[1*p + (k_offset+0)][start_rank + 0] = -1
        W[(k_offset+0)*n + 0][start_rank + 0] = 1 # C00
        
        # M2: (A00+A10)B00
        U[0][start_rank + 1] = 1
        U[1*m + 0][start_rank + 1] = 1
        V[0*p + (k_offset+0)][start_rank + 1] = 1
        W[(k_offset+0)*n + 1][start_rank + 1] = 1 # C10
        
        # M3: (A10-A00)B01
        U[1*m + 0][start_rank + 2] = 1
        U[0][start_rank + 2] = -1
        V[0*p + (k_offset+1)][start_rank + 2] = 1
        W[(k_offset+1)*n + 1][start_rank + 2] = 1 # C11
        
        # M4: A11(B00-B10)
        U[1*m + 1][start_rank + 3] = 1
        V[0*p + (k_offset+0)][start_rank + 3] = 1
        V[1*p + (k_offset+0)][start_rank + 3] = -1
        W[(k_offset+0)*n + 1][start_rank + 3] = 1 # C10
        
        # M5: (A00+A01)B11
        U[0][start_rank + 4] = 1
        U[0*m + 1][start_rank + 4] = 1
        V[1*p + (k_offset+1)][start_rank + 4] = 1
        W[(k_offset+1)*n + 0][start_rank + 4] = 1 # C01
        
        # M6: (A11-A01)(B01+B11)
        U[1*m + 1][start_rank + 5] = 1
        U[0*m + 1][start_rank + 5] = -1
        V[0*p + (k_offset+1)][start_rank + 5] = 1
        V[1*p + (k_offset+1)][start_rank + 5] = 1
        W[(k_offset+1)*n + 1][start_rank + 5] = 1 # C11
        W[(k_offset+1)*n + 0][start_rank + 5] = -1 # C01
        
        # M7: (A01-A11)(B00+B10)
        U[0*m + 1][start_rank + 6] = 1
        U[1*m + 1][start_rank + 6] = -1
        V[0*p + (k_offset+0)][start_rank + 6] = 1
        V[1*p + (k_offset+0)][start_rank + 6] = 1
        W[(k_offset+0)*n + 0][start_rank + 6] = 1 # C00
        W[(k_offset+0)*n + 1][start_rank + 6] = -1 # C10

    # Block 1: Columns 0,1
    add_strassen_222(0, 0)
    
    # Block 2: Columns 2,3
    add_strassen_222(7, 2)
    
    # Block 3: Column 4 (2x2 by 2x1)
    # Rank 4 schoolbook
    start_rank = 14
    k_offset = 4
    # C00 = A00 B00 + A01 B10
    U[0][start_rank + 0] = 1
    V[0*p + k_offset][start_rank + 0] = 1
    W[k_offset*n + 0][start_rank + 0] = 1
    
    U[1*m + 1][start_rank + 1] = 1 # A01? No, A01 is idx 1. Wait.
    # A indices: 0:00, 1:01, 2:10, 3:11.
    # C0 = A00 B00 + A01 B10.
    # Term 1: A00 B00. U[0]=1, V[0*5+4]=1, W[4*2+0]=1.
    # Term 2: A01 B10. U[1]=1, V[1*5+4]=1, W[4*2+0]=1.
    U[0][start_rank + 0] = 1
    V[0*p + k_offset][start_rank + 0] = 1
    W[k_offset*n + 0][start_rank + 0] = 1
    
    U[0*m + 1][start_rank + 1] = 1 # A01
    V[1*p + k_offset][start_rank + 1] = 1 # B10
    W[k_offset*n + 0][start_rank + 1] = 1
    
    # C1 = A10 B00 + A11 B10
    # Term 3: A10 B00. U[2]=1, V[0*5+4]=1, W[4*2+1]=1.
    # Term 4: A11 B10. U[3]=1, V[1*5+4]=1, W[4*2+1]=1.
    U[1*m + 0][start_rank + 2] = 1 # A10
    V[0*p + k_offset][start_rank + 2] = 1
    W[k_offset*n + 1][start_rank + 2] = 1
    
    U[1*m + 1][start_rank + 3] = 1 # A11
    V[1*p + k_offset][start_rank + 3] = 1
    W[k_offset*n + 1][start_rank + 3] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
