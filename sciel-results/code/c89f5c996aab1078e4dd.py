"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the p dimension (13) into 6 blocks of size 2 and 1 block of size 1.
    # For each 2x2x2 block, we use Strassen (rank 7).
    # For the 2x2x1 block, we use schoolbook (rank 4).
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 tensor
    def add_tensor(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Indices for A: i*m + j -> 0,1,2,3
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # Indices for B: j*p + k
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> p
    # j=1, k=1 -> p+1
    # ...
    # j=0, k=k0 -> k0
    # j=1, k=k0 -> p + k0
    
    # Indices for C: k*n + i
    # k=0, i=0 -> 0
    # k=1, i=0 -> 1
    # k=0, i=1 -> n
    # k=1, i=1 -> n+1
    # ...
    # k=k0, i=0 -> k0*n
    # k=k0, i=1 -> k0*n + 1

    # Process 6 blocks of size 2
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # B indices for this block
        # B[0, k0], B[0, k1], B[1, k0], B[1, k1]
        # V indices:
        # v_00 = 0*p + k0 = k0
        # v_01 = 0*p + k1 = k1
        # v_10 = 1*p + k0 = p + k0
        # v_11 = 1*p + k1 = p + k1
        
        # C indices for this block
        # C[0, k0], C[0, k1], C[1, k0], C[1, k1]
        # W indices:
        # w_00 = k0*n + 0 = 2*k0
        # w_01 = k1*n + 0 = 2*k1
        # w_10 = k0*n + 1 = 2*k0 + 1
        # w_11 = k1*n + 1 = 2*k1 + 1
        
        # Strassen factors for A (same for all blocks)
        # u1 = A00 + A11
        # u2 = A10 + A11
        # u3 = A00
        # u4 = A11
        # u5 = A00 + A01
        # u6 = A01 - A10
        # u7 = A10 - A00
        
        # V factors for this block
        # v1 = B00 + B11
        # v2 = B01
        # v3 = B10 - B11
        # v4 = B01 - B00
        # v5 = B11
        # v6 = B00 + B01
        # v7 = B11 + B10
        
        # W factors for this block
        # w1: C00 += 1, C11 += 1
        # w2: C10 += 1
        # w3: C01 += 1, C11 -= 1
        # w4: C00 += 1, C10 += 1
        # w5: C01 += 1, C00 -= 1
        # w6: C11 += 1
        # w7: C00 += 1, C01 += 1
        
        # Construct vectors (size n*m=4 for U, m*p=26 for V, p*n=26 for W)
        
        # U vectors
        u1 = [0]*4; u1[0]=1; u1[3]=1
        u2 = [0]*4; u2[2]=1; u2[3]=1
        u3 = [0]*4; u3[0]=1
        u4 = [0]*4; u4[3]=1
        u5 = [0]*4; u5[0]=1; u5[1]=1
        u6 = [0]*4; u6[1]=1; u6[2]=-1
        u7 = [0]*4; u7[2]=1; u7[0]=-1
        
        # V vectors (local to block, need to map to global V indices)
        # Global V index = j*p + k
        def get_v(j, k):
            return j*p + k
        
        v1 = [0]*(m*p)
        v1[get_v(0, k0)] = 1
        v1[get_v(1, k1)] = 1
        
        v2 = [0]*(m*p)
        v2[get_v(0, k1)] = 1
        
        v3 = [0]*(m*p)
        v3[get_v(1, k0)] = 1
        v3[get_v(1, k1)] = -1
        
        v4 = [0]*(m*p)
        v4[get_v(0, k1)] = 1
        v4[get_v(0, k0)] = -1
        
        v5 = [0]*(m*p)
        v5[get_v(1, k1)] = 1
        
        v6 = [0]*(m*p)
        v6[get_v(0, k0)] = 1
        v6[get_v(0, k1)] = 1
        
        v7 = [0]*(m*p)
        v7[get_v(1, k1)] = 1
        v7[get_v(1, k0)] = 1
        
        # W vectors (local to block, need to map to global W indices)
        # Global W index = k*n + i
        def get_w(k, i):
            return k*n + i
            
        w1 = [0]*(p*n)
        w1[get_w(k0, 0)] = 1
        w1[get_w(k1, 1)] = 1
        
        w2 = [0]*(p*n)
        w2[get_w(k0, 1)] = 1
        
        w3 = [0]*(p*n)
        w3[get_w(k1, 0)] = 1
        w3[get_w(k1, 1)] = -1
        
        w4 = [0]*(p*n)
        w4[get_w(k0, 0)] = 1
        w4[get_w(k0, 1)] = 1
        
        w5 = [0]*(p*n)
        w5[get_w(k1, 0)] = 1
        w5[get_w(k0, 0)] = -1
        
        w6 = [0]*(p*n)
        w6[get_w(k1, 1)] = 1
        
        w7 = [0]*(p*n)
        w7[get_w(k0, 0)] = 1
        w7[get_w(k1, 0)] = 1
        
        # Add to columns
        add_tensor(u1, v1, w1)
        add_tensor(u2, v2, w2)
        add_tensor(u3, v3, w3)
        add_tensor(u4, v4, w4)
        add_tensor(u5, v5, w5)
        add_tensor(u6, v6, w6)
        add_tensor(u7, v7, w7)
        
    # Process the last block of size 1 (k=12)
    k0 = 12
    
    # Schoolbook for 2x2x1
    # M1 = A00 B0k
    # M2 = A01 B1k
    # M3 = A10 B0k
    # M4 = A11 B1k
    # C0k = M1 + M2
    # C1k = M3 + M4
    
    u1 = [0]*4; u1[0]=1
    u2 = [0]*4; u2[1]=1
    u3 = [0]*4; u3[2]=1
    u4 = [0]*4; u4[3]=1
    
    v1 = [0]*(m*p)
    v1[get_v(0, k0)] = 1
    
    v2 = [0]*(m*p)
    v2[get_v(1, k0)] = 1
    
    v3 = [0]*(m*p)
    v3[get_v(0, k0)] = 1
    
    v4 = [0]*(m*p)
    v4[get_v(1, k0)] = 1
    
    w1 = [0]*(p*n)
    w1[get_w(k0, 0)] = 1
    
    w2 = [0]*(p*n)
    w2[get_w(k0, 0)] = 1
    
    w3 = [0]*(p*n)
    w3[get_w(k0, 1)] = 1
    
    w4 = [0]*(p*n)
    w4[get_w(k0, 1)] = 1
    
    add_tensor(u1, v1, w1)
    add_tensor(u2, v2, w2)
    add_tensor(u3, v3, w3)
    add_tensor(u4, v4, w4)
    
    rank = len(U_cols)
    
    # Convert lists of lists to proper matrix format if needed, but task accepts lists of lists
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [ [0]*rank for _ in range(n*m) ]
    V = [ [0]*rank for _ in range(m*p) ]
    W = [ [0]*rank for _ in range(p*n) ]
    
    for c in range(rank):
        for i in range(n*m):
            U[i][c] = U_cols[c][i]
        for j in range(m*p):
            V[j][c] = V_cols[c][j]
        for k in range(p*n):
            W[k][c] = W_cols[c][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
