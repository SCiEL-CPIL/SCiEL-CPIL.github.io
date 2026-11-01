import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Strategy: Block decomposition with Strassen for 2x2 blocks
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 3 Strassen ops (rank 7 each) = 21
    # A2*B2: 3 schoolbook 2x3*3x2 ops (rank 12 each) = 36
    # Total: 57
    
    # First, verify Strassen correctness
    def verify_strassen():
        import random
        random.seed(123)
        for _ in range(200):
            a,b,c,d = [random.randint(-10,10) for _ in range(4)]
            e,f,g,h = [random.randint(-10,10) for _ in range(4)]
            # Standard Strassen
            M1=(a+d)*(e+g); M2=(c+d)*e; M3=a*(f-g); M4=d*(h-g)
            M5=(a+b)*g; M6=(c+d)*(h-f); M7=(a+b)*(f-h)
            C00=M1-M2-M4+M3; C01=M3+M4+M7; C10=M2+M5+M6; C11=M1-M5+M6+M7
            if C00 != a*e+b*g or C01 != a*f+b*h or C10 != c*e+d*g or C11 != c*f+d*h:
                return False
        return True
    
    use_strassen = verify_strassen()
    
    if not use_strassen:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i][col] += val
        col += 1
    
    def add_strassen_mapped(u_map, v_map, w_map):
        """Add Strassen 2x2*2x2 with index mappings. u_map,v_map,w_map are 4-element lists."""
        nonlocal col
        # M1=(a+d)(e+g)
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[0]]=1; u[u_map[3]]=1
        v[v_map[0]]=1; v[v_map[2]]=1
        w[w_map[0]]=1; w[w_map[3]]=1
        add_term(u,v,w)
        # M2=(c+d)e
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[2]]=1; u[u_map[3]]=1
        v[v_map[0]]=1
        w[w_map[2]]=1
        add_term(u,v,w)
        # M3=a(f-g)
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[0]]=1
        v[v_map[1]]=1; v[v_map[2]]=-1
        w[w_map[0]]=1; w[w_map[1]]=1
        add_term(u,v,w)
        # M4=d(h-g)
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[3]]=1
        v[v_map[2]]=-1; v[v_map[3]]=1
        w[w_map[0]]=-1; w[w_map[1]]=1
        add_term(u,v,w)
        # M5=(a+b)g
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[0]]=1; u[u_map[1]]=1
        v[v_map[2]]=1
        w[w_map[2]]=1; w[w_map[3]]=-1
        add_term(u,v,w)
        # M6=(c+d)(h-f)
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[2]]=1; u[u_map[3]]=1
        v[v_map[1]]=-1; v[v_map[3]]=1
        w[w_map[2]]=1; w[w_map[3]]=1
        add_term(u,v,w)
        # M7=(a+b)(f-h)
        u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
        u[u_map[0]]=1; u[u_map[1]]=1
        v[v_map[1]]=1; v[v_map[3]]=-1
        w[w_map[1]]=1; w[w_map[3]]=1
        add_term(u,v,w)
    
    def add_2x3_3x2_mapped(u_map, v_map, w_map):
        """Add schoolbook 2x3*3x2. u_map:6, v_map:6, w_map:4"""
        nonlocal col
        for k in range(3):
            u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
            u[u_map[k]]=1; v[v_map[k*2]]=1; w[w_map[0]]=1
            add_term(u,v,w)
            u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
            u[u_map[k]]=1; v[v_map[k*2+1]]=1; w[w_map[1]]=1
            add_term(u,v,w)
            u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
            u[u_map[3+k]]=1; v[v_map[k*2]]=1; w[w_map[2]]=1
            add_term(u,v,w)
            u=[0]*(n*m); v=[0]*(m*p); w=[0]*(p*n)
            u[u_map[3+k]]=1; v[v_map[k*2+1]]=1; w[w_map[3]]=1
            add_term(u,v,w)
    
    # U indices: A[i,j] at i*m+j: A00=0,A01=1,A02=2,A03=3,A04=4,A10=5,A11=6,A12=7,A13=8,A14=9
    # V indices: B[j,k] at j*p+k: B00=0,...,B05=5,B10=6,...,B15=11,B20=12,...,B25=17,B30=18,...,B35=23,B40=24,...,B45=29
    # W indices: C[i,k] at k*n+i: C00=0,C10=1,C01=2,C11=3,C02=4,C12=5,C03=6,C13=7,C04=8,C14=9,C05=10,C15=11
    
    # A1 = A[:,0:2] -> U: [A00,A01,A10,A11] = [0,1,5,6]
    a1_u = [0, 1, 5, 6]
    
    # B1 = B[0:2,:] -> V: B00-B15 = indices 0-11
    # B1[:,0:2] -> [B00,B01,B10,B11] = [0,1,6,7]
    # B1[:,2:4] -> [B02,B03,B12,B13] = [2,3,8,9]
    # B1[:,4:6] -> [B04,B05,B14,B15] = [4,5,10,11]
    b1_v = [[0,1,6,7], [2,3,8,9], [4,5,10,11]]
    
    # C blocks (2x2 each):
    # C[:,0:2] -> W: C00=0,C10=1,C01=2,C11=3 -> w_map=[0,2,1,3] (C00,C01,C10,C11)
    # C[:,2:4] -> W: C02=4,C12=5,C03=6,C13=7 -> w_map=[4,6,5,7]
    # C[:,4:6] -> W: C04=8,C14=9,C05=10,C15=11 -> w_map=[8,10,9,11]
    c_w = [[0,2,1,3], [4,6,5,7], [8,10,9,11]]
    
    # Add A1*B1 using Strassen
    for blk in range(3):
        add_strassen_mapped(a1_u, b1_v[blk], c_w[blk])
    
    # A2 = A[:,2:5] -> U: [A02,A03,A04,A12,A13,A14] = [2,3,4,7,8,9]
    a2_u = [2, 3, 4, 7, 8, 9]
    
    # B2 = B[2:5,:] -> V: B20-B45 = indices 12-29
    # B2[:,0:2] -> [B20,B21,B30,B31,B40,B41] = [12,13,18,19,24,25]
    # B2[:,2:4] -> [B22,B23,B32,B33,B42,B43] = [14,15,20,21,26,27]
    # B2[:,4:6] -> [B24,B25,B34,B35,B44,B45] = [16,17,22,23,28,29]
    b2_v = [[12,13,18,19,24,25], [14,15,20,21,26,27], [16,17,22,23,28,29]]
    
    # Add A2*B2 using schoolbook
    for blk in range(3):
        add_2x3_3x2_mapped(a2_u, b2_v[blk], c_w[blk])
    
    # Verify the full decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    # Fall back to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][column] = 1
                                V[jj * p + kk][column] = 1
                                W[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
