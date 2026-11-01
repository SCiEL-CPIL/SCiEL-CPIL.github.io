import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 2, 14
    rank = 49
    
    # Strassen's 7 linear forms for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # U_forms[s][i] = coefficient of A[i] in the s-th Strassen form
    # A indices: 0=a, 1=b, 2=c, 3=d
    U_forms = [
        [1, 0, 0, 1],   # M1 = a+d
        [0, 0, 1, 1],   # M2 = c+d
        [1, 0, 0, 0],   # M3 = a
        [0, 0, 0, 1],   # M4 = d
        [1, 1, 0, 0],   # M5 = a+b
        [-1, 0, 1, 0],  # M6 = c-a
        [0, 1, 0, -1],  # M7 = b-d
    ]
    
    # For each block g (0..6), the B and C indices are:
    # B: e=B[0,2g], f=B[0,2g+1], g_=B[1,2g], h=B[1,2g+1]
    # C: C[0,2g], C[0,2g+1], C[1,2g], C[1,2g+1]
    # 
    # Strassen terms:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g_-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g_+h)
    #
    # C[0,2g] = M1 + M4 - M5 + M7
    # C[0,2g+1] = M3 + M5
    # C[1,2g] = M2 + M4
    # C[1,2g+1] = M1 - M2 + M3 + M6
    
    # V_forms[s][local_j, local_k] = coefficient of B[local_j, local_k] in the s-th form
    # local_j: 0=e, 1=g_; local_k: 0=first col, 1=second col
    # So: (0,0)=e, (0,1)=f, (1,0)=g_, (1,1)=h
    V_forms = [
        [[1, 0], [0, 1]],   # M1: e+h
        [[1, 0], [0, 0]],   # M2: e
        [[0, 1], [0, -1]],  # M3: f-h
        [[-1, 0], [1, 0]],  # M4: g_-e
        [[0, 0], [0, 1]],   # M5: h
        [[1, 1], [0, 0]],   # M6: e+f
        [[0, 0], [1, 1]],   # M7: g_+h
    ]
    
    # W_forms[s][out_i, out_k] = coefficient for C[out_i, out_k] in the s-th term
    # out_i: 0,1; out_k: 0=first col, 1=second col
    W_forms = [
        [[1, 0], [0, 1]],   # M1: C[0,0]+1, C[1,1]+1
        [[0, 0], [1, -1]],  # M2: C[1,0]+1, C[1,1]-1
        [[0, 1], [0, 1]],   # M3: C[0,1]+1, C[1,1]+1
        [[1, 0], [1, 0]],   # M4: C[0,0]+1, C[1,0]+1
        [[-1, 1], [0, 0]],  # M5: C[0,0]-1, C[0,1]+1
        [[0, 0], [0, 1]],   # M6: C[1,1]+1
        [[1, 0], [0, 0]],   # M7: C[0,0]+1
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for g in range(7):
        if time.time() > deadline:
            break
        for s in range(7):
            col = g * 7 + s
            
            # Fill U
            for i in range(4):
                U[i][col] = U_forms[s][i]
            
            # Fill V
            # B[0, 2g] -> V[0*14 + 2g] = V[2g]
            # B[0, 2g+1] -> V[0*14 + 2g+1] = V[2g+1]
            # B[1, 2g] -> V[1*14 + 2g] = V[14+2g]
            # B[1, 2g+1] -> V[1*14 + 2g+1] = V[14+2g+1]
            V[2*g][col] = V_forms[s][0][0]
            V[2*g+1][col] = V_forms[s][0][1]
            V[14+2*g][col] = V_forms[s][1][0]
            V[14+2*g+1][col] = V_forms[s][1][1]
            
            # Fill W
            # C[0, 2g] -> W[2g*2+0] = W[4g]
            # C[0, 2g+1] -> W[(2g+1)*2+0] = W[4g+2]
            # C[1, 2g] -> W[2g*2+1] = W[4g+1]
            # C[1, 2g+1] -> W[(2g+1)*2+1] = W[4g+3]
            W[4*g][col] = W_forms[s][0][0]
            W[4*g+2][col] = W_forms[s][0][1]
            W[4*g+1][col] = W_forms[s][1][0]
            W[4*g+3][col] = W_forms[s][1][1]
    
    return {"U": U, "V": V, "W": W}
