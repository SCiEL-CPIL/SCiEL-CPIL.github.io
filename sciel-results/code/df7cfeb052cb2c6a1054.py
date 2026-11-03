"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u, v, w):
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
    
    # Strassen blocks: A[0:2,0:2] * B[0:2, 2s:2s+2] for s=0,1,2,3
    strassen_handled = set()
    for s in range(4):
        for i in range(2):
            for j in range(2):
                for k in range(2*s, 2*s+2):
                    strassen_handled.add((i, j, k))
    
    for s in range(4):
        if time.time() > DEADLINE:
            break
        k0 = 2*s
        k1 = 2*s+1
        
        # m1 = (a+d)(e+h): U: A[0,0]+A[1,0], V: B[0,k0]+B[0,k1], W: C[0,k0]+=1, C[1,k1]+=1
        u1 = [0.0]*(n*m); u1[0]=1; u1[3]=1
        v1 = [0.0]*(m*p); v1[0*8+k0]=1; v1[0*8+k1]=1
        w1 = [0.0]*(p*n); w1[k0*3+0]=1; w1[k1*3+1]=1
        add_term(u1,v1,w1)
        
        # m2 = (c+d)e: U: A[1,0]+A[1,1], V: B[0,k0], W: C[1,k0]+=1, C[1,k1]+=-1
        u2 = [0.0]*(n*m); u2[3]=1; u2[4]=1
        v2 = [0.0]*(m*p); v2[0*8+k0]=1
        w2 = [0.0]*(p*n); w2[k0*3+1]=1; w2[k1*3+1]=-1
        add_term(u2,v2,w2)
        
        # m3 = a(f-h): U: A[0,0], V: B[0,k1]-B[1,k1], W: C[0,k1]+=1, C[1,k1]+=1
        u3 = [0.0]*(n*m); u3[0]=1
        v3 = [0.0]*(m*p); v3[0*8+k1]=1; v3[1*8+k1]=-1
        w3 = [0.0]*(p*n); w3[k1*3+0]=1; w3[k1*3+1]=1
        add_term(u3,v3,w3)
        
        # m4 = d(g-e): U: A[1,1], V: B[1,k0]-B[0,k0], W: C[0,k0]+=1, C[1,k0]+=1
        u4 = [0.0]*(n*m); u4[4]=1
        v4 = [0.0]*(m*p); v4[1*8+k0]=1; v4[0*8+k0]=-1
        w4 = [0.0]*(p*n); w4[k0*3+0]=1; w4[k0*3+1]=1
        add_term(u4,v4,w4)
        
        # m5 = (a+b)h: U: A[0,0]+A[0,1], V: B[1,k1], W: C[0,k1]+=1, C[0,k0]+=-1
        u5 = [0.0]*(n*m); u5[0]=1; u5[1]=1
        v5 = [0.0]*(m*p); v5[1*8+k1]=1
        w5 = [0.0]*(p*n); w5[k1*3+0]=1; w5[k0*3+0]=-1
        add_term(u5,v5,w5)
        
        # m6 = (c-a)(e+f): U: A[1,0]-A[0,0], V: B[0,k0]+B[0,k1], W: C[1,k1]+=1
        u6 = [0.0]*(n*m); u6[3]=1; u6[0]=-1
        v6 = [0.0]*(m*p); v6[0*8+k0]=1; v6[0*8+k1]=1
        w6 = [0.0]*(p*n); w6[k1*3+1]=1
        add_term(u6,v6,w6)
        
        # m7 = (b-d)(g+h): U: A[0,1]-A[1,1], V: B[1,k0]+B[1,k1], W: C[0,k0]+=1
        u7 = [0.0]*(n*m); u7[1]=1; u7[4]=-1
        v7 = [0.0]*(m*p); v7[1*8+k0]=1; v7[1*8+k1]=1
        w7 = [0.0]*(p*n); w7[k0*3+0]=1
        add_term(u7,v7,w7)
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_handled:
                    continue
                u = [0.0]*(n*m); u[i*m+j] = 1.0
                v = [0.0]*(m*p); v[j*p+k] = 1.0
                w = [0.0]*(p*n); w[k*n+i] = 1.0
                add_term(u,v,w)
    
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
