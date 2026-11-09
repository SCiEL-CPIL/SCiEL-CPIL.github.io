import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of columns (rank vectors)
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2x2
    # A is 2x2, B is 2x2
    # u_vec index: i*2+j
    # v_vec index: j*2+k
    # w_vec index: k*2+i
    
    def strassen_2x2x2(offset_u, offset_v, offset_w, scale=1):
        # 7 products
        # M1 = (A00+A11)(B00+B11)
        u1 = [0]*4; u1[0]=1; u1[3]=1
        v1 = [0]*4; v1[0]=1; v1[3]=1
        w1 = [0]*4; w1[0]=1; w1[3]=1 # C00, C11
        # Wait, C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So w1 for M1 is 1 at C00 and C11.
        # Index for C00 is 0*2+0=0. Index for C11 is 1*2+1=3.
        
        # M2 = (A10+A11)B00
        u2 = [0]*4; u2[2]=1; u2[3]=1
        v2 = [0]*4; v2[0]=1
        w2 = [0]*4; w2[2]=-1 # C10 index 1*2+0=2. C11 index 3.
        # C10 = M2 + M4
        # C11 = M1 - M2 + ...
        # So w2 has -1 at C11? No, C11 = M1 - M2 + M3 + M6.
        # So w2[3] = -1. w2[2] = 1.
        
        # M3 = A00(B01-B11)
        u3 = [0]*4; u3[0]=1
        v3 = [0]*4; v3[1]=1; v3[3]=-1
        w3 = [0]*4; w3[1]=1; w3[3]=1 # C01 index 0*2+1=1. C11 index 3.
        
        # M4 = A11(B10-B00)
        u4 = [0]*4; u4[3]=1
        v4 = [0]*4; v4[2]=1; v4[0]=-1
        w4 = [0]*4; w4[0]=1; w4[2]=1 # C00 index 0. C10 index 2.
        
        # M5 = (A00+A01)B11
        u5 = [0]*4; u5[0]=1; u5[1]=1
        v5 = [0]*4; v5[3]=1
        w5 = [0]*4; w5[0]=-1; w5[1]=1 # C00 index 0. C01 index 1.
        
        # M6 = (A01-A10)(B00+B01)
        u6 = [0]*4; u6[1]=1; u6[2]=-1
        v6 = [0]*4; v6[0]=1; v6[1]=1
        w6 = [0]*4; w6[3]=1 # C11 index 3.
        
        # M7 = (A10-A00)(B11+B00)
        u7 = [0]*4; u7[2]=1; u7[0]=-1
        v7 = [0]*4; v7[3]=1; v7[0]=1
        w7 = [0]*4; w7[0]=1 # C00 index 0.
        
        # Apply offsets and scale
        # Note: offsets are for the global U, V, W indices.
        # U index: i*m+j. For 2x2 block at i0, j0: (i0+i)*m + (j0+j).
        # V index: j*p+k. For 2x2 block at j0, k0: (j0+j)*p + (k0+k).
        # W index: k*n+i. For 2x2 block at k0, i0: (k0+k)*n + (i0+i).
        
        def apply_offset(vec, offset_func):
            new_vec = [0] * (n*m if 'u' in offset_func else m*p if 'v' in offset_func else p*n)
            # This is getting complicated. Let's just use full size vectors.
            return vec
            
        # Actually, it's easier to just create full size vectors directly.
        # Let's rewrite add_term to take full size vectors.
        
        # We will create full size vectors for each Strassen term.
        # Size U: n*m = 10
        # Size V: m*p = 30
        # Size W: p*n = 12
        
        # Block 1: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
        # i0=0, j0=0, k0=0
        # U indices: (0+i)*5 + (0+j)
        # V indices: (0+j)*6 + (0+k)
        # W indices: (0+k)*2 + (0+i)
        
        # We will just call a function that takes i0, j0, k0 and adds the 7 terms.
        
    # Let's implement a function to add a 2x2x2 Strassen block at position (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        # 7 terms
        terms = []
        # M1
        u = [0]*10; u[i0*5+j0]=1; u[(i0+1)*5+(j0+1)]=1
        v = [0]*30; v[j0*6+k0]=1; v[(j0+1)*6+(k0+1)]=1
        w = [0]*12; w[k0*2+i0]=1; w[(k0+1)*2+(i0+1)]=1
        terms.append((u,v,w))
        
        # M2
        u = [0]*10; u[(i0+1)*5+j0]=1; u[(i0+1)*5+(j0+1)]=1
        v = [0]*30; v[j0*6+k0]=1
        w = [0]*12; w[k0*2+(i0+1)]=1; w[(k0+1)*2+(i0+1)]=-1
        terms.append((u,v,w))
        
        # M3
        u = [0]*10; u[i0*5+j0]=1
        v = [0]*30; v[j0*6+(k0+1)]=1; v[(j0+1)*6+(k0+1)]=-1
        w = [0]*12; w[(k0+1)*2+i0]=1; w[(k0+1)*2+(i0+1)]=1
        terms.append((u,v,w))
        
        # M4
        u = [0]*10; u[(i0+1)*5+(j0+1)]=1
        v = [0]*30; v[(j0+1)*6+k0]=1; v[j0*6+k0]=-1
        w = [0]*12; w[k0*2+i0]=1; w[k0*2+(i0+1)]=1
        terms.append((u,v,w))
        
        # M5
        u = [0]*10; u[i0*5+j0]=1; u[i0*5+(j0+1)]=1
        v = [0]*30; v[(j0+1)*6+(k0+1)]=1
        w = [0]*12; w[k0*2+i0]=-1; w[(k0+1)*2+i0]=1
        terms.append((u,v,w))
        
        # M6
        u = [0]*10; u[i0*5+(j0+1)]=1; u[(i0+1)*5+j0]=-1
        v = [0]*30; v[j0*6+k0]=1; v[j0*6+(k0+1)]=1
        w = [0]*12; w[(k0+1)*2+(i0+1)]=1
        terms.append((u,v,w))
        
        # M7
        u = [0]*10; u[(i0+1)*5+j0]=1; u[i0*5+j0]=-1
        v = [0]*30; v[(j0+1)*6+(k0+1)]=1; v[j0*6+k0]=1
        w = [0]*12; w[k0*2+i0]=1
        terms.append((u,v,w))
        
        for u,v,w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
            
    # Block 1: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # Block 2: A[0:2, 0:2], B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # Block 3: A[0:2, 0:2], B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 0, 4)
    
    # This covers A1 (2x2) * B1 (2x6). Rank 21.
    
    # Now A2 (2x2) * B2 (2x6).
    # A2 is A[0:2, 2:4]. B2 is B[2:4, 0:6].
    # i0=0, j0=2.
    # k0=0, 2, 4.
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # This covers A2 * B2. Rank 21.
    
    # Now A3 (2x1) * B3 (1x6).
    # A3 is A[0:2, 4:5]. B3 is B[4:5, 0:6].
    # i0=0, j0=4.
    # k0=0..5.
    # Rank 12.
    # C[i,k] = A[i,4] B[4,k].
    # We can use 12 rank-1 terms.
    # For each i in {0,1}, k in {0..5}:
    # Term: U[i*5+4]=1, V[4*6+k]=1, W[k*2+i]=1.
    for i in range(2):
        for k in range(6):
            u = [0]*10; u[i*5+4]=1
            v = [0]*30; v[4*6+k]=1
            w = [0]*12; w[k*2+i]=1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
            
    # Total rank: 21 + 21 + 12 = 54.
    
    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
