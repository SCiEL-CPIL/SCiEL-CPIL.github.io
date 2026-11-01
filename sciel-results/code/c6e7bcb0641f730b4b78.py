import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 by 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-g)
    # m4 = d(g-h)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(f+h)
    # C00 = m1 + m4 - m6 - m7
    # C01 = m3 + m7
    # C10 = m2 + m6
    # C11 = m1 - m2 + m3 + m5
    
    # For each of the 5 blocks (columns 2t, 2t+1):
    # Block t: A is the same 2x2, B_t = [[B0,2t, B0,2t+1], [B1,2t, B1,2t+1]]
    
    # U factors (A-linear forms), same for all blocks:
    # r=0: a+d  -> U[0,0]=1, U[1,1]=1  (a=A[0,0], d=A[1,1])
    # r=1: c+d  -> U[2,0]=1, U[3,1]=1  (c=A[1,0], d=A[1,1])  Wait, U is indexed by i*m+j
    #     i=0,j=0 -> index 0; i=0,j=1 -> index 1; i=1,j=0 -> index 2; i=1,j=1 -> index 3
    # r=0: a+d = A[0,0]+A[1,1] -> U[0]=1, U[3]=1
    # r=1: c+d = A[1,0]+A[1,1] -> U[2]=1, U[3]=1
    # r=2: a = A[0,0] -> U[0]=1
    # r=3: d = A[1,1] -> U[3]=1
    # r=4: a+b = A[0,0]+A[0,1] -> U[0]=1, U[1]=1
    # r=5: c-a = A[1,0]-A[0,0] -> U[2]=1, U[0]=-1
    # r=6: b-d = A[0,1]-A[1,1] -> U[1]=1, U[3]=-1
    
    # For block t (columns 2t, 2t+1):
    # e = B[0,2t], f = B[0,2t+1], g = B[1,2t], h = B[1,2t+1]
    # V is indexed by j*p+k: j=0,k=0 -> 0; j=0,k=1 -> 1; ...; j=1,k=0 -> 10; j=1,k=1 -> 11; ...
    # r=0: e+h = B[0,2t]+B[1,2t+1] -> V[2t]=1, V[10+2t+1]=1
    # r=1: e = B[0,2t] -> V[2t]=1
    # r=2: f-g = B[0,2t+1]-B[1,2t] -> V[2t+1]=1, V[10+2t]=-1
    # r=3: g-h = B[1,2t]-B[1,2t+1] -> V[10+2t]=1, V[10+2t+1]=-1
    # r=4: h = B[1,2t+1] -> V[10+2t+1]=1
    # r=5: e+f = B[0,2t]+B[0,2t+1] -> V[2t]=1, V[2t+1]=1
    # r=6: f+h = B[0,2t+1]+B[1,2t+1] -> V[2t+1]=1, V[10+2t+1]=1
    
    # W factors (output routing), W is indexed by k*n+i: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; ...
    # For block t, output columns are 2t and 2t+1:
    # C[0,2t] = m1 + m4 - m6 - m7 -> W[2t*2+0] = W[4t] gets contributions from r=0(+1), r=3(+1), r=5(-1), r=6(-1)
    # C[0,2t+1] = m3 + m7 -> W[4t+1] gets r=2(+1), r=6(+1)
    # C[1,2t] = m2 + m6 -> W[4t+2] gets r=1(+1), r=5(+1)
    # C[1,2t+1] = m1 - m2 + m3 + m5 -> W[4t+3] gets r=0(+1), r=1(-1), r=2(+1), r=4(+1)
    
    rank = 35
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    for t in range(5):
        base = t * 7
        # r=0: m1
        U[0][base+0] = 1  # a
        U[3][base+0] = 1  # d
        V[2*t][base+0] = 1  # e
        V[10+2*t+1][base+0] = 1  # h
        W[4*t+0][base+0] = 1  # C[0,2t] += m1
        W[4*t+3][base+0] = 1  # C[1,2t+1] += m1
        
        # r=1: m2
        U[2][base+1] = 1  # c
        U[3][base+1] = 1  # d
        V[2*t][base+1] = 1  # e
        W[4*t+2][base+1] = 1  # C[1,2t] += m2
        W[4*t+3][base+1] = -1  # C[1,2t+1] -= m2
        
        # r=2: m3
        U[0][base+2] = 1  # a
        V[2*t+1][base+2] = 1  # f
        V[10+2*t][base+2] = -1  # -g
        W[4*t+1][base+2] = 1  # C[0,2t+1] += m3
        W[4*t+3][base+2] = 1  # C[1,2t+1] += m3
        
        # r=3: m4
        U[3][base+3] = 1  # d
        V[10+2*t][base+3] = 1  # g
        V[10+2*t+1][base+3] = -1  # -h
        W[4*t+0][base+3] = 1  # C[0,2t] += m4
        
        # r=4: m5
        U[0][base+4] = 1  # a
        U[1][base+4] = 1  # b
        V[10+2*t+1][base+4] = 1  # h
        W[4*t+3][base+4] = 1  # C[1,2t+1] += m5
        
        # r=5: m6
        U[2][base+5] = 1  # c
        U[0][base+5] = -1  # -a
        V[2*t][base+5] = 1  # e
        V[2*t+1][base+5] = 1  # f
        W[4*t+2][base+5] = 1  # C[1,2t] += m6
        W[4*t+0][base+5] = -1  # C[0,2t] -= m6
        
        # r=6: m7
        U[1][base+6] = 1  # b
        U[3][base+6] = -1  # -d
        V[2*t+1][base+6] = 1  # f
        V[10+2*t+1][base+6] = 1  # h
        W[4*t+1][base+6] = 1  # C[0,2t+1] += m7
        W[4*t+0][base+6] = -1  # C[0,2t] -= m7
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    for trial in range(5):
        A_test = np.random.randn(2, 2)
        B_test = np.random.randn(2, 10)
        C_expected = A_test @ B_test
        
        # Compute via decomposition
        C_computed = np.zeros((2, 10))
        for r in range(rank):
            f_r = np.dot(U_np[:, r], A_test.flatten())
            g_r = np.dot(V_np[:, r], B_test.flatten())
            for k in range(10):
                for i in range(2):
                    C_computed[i, k] += W_np[k*2+i, r] * f_r * g_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            rank = 40
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(20)]
            W = [[0]*rank for _ in range(20)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(10):
                        U[i*2+j][col] = 1
                        V[j*10+k][col] = 1
                        W[k*2+i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
