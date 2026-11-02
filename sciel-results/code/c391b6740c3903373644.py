import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U_t, V_t, W_t are 2x2 matrices
    # t=0 corresponds to M1, t=1 to M2, etc.
    
    # M1 = (a+d)(e+h)
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 1]]
    
    # M2 = (c+d)e
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 0], [1, -1]]
    
    # M3 = a(f-h)
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 1], [0, -1]]
    W3 = [[0, 1], [0, 1]]
    
    # M4 = d(g-e)
    U4 = [[0, 0], [0, 1]]
    V4 = [[-1, 0], [1, 0]]
    W4 = [[1, 0], [1, 0]]
    
    # M5 = (a+b)h
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 1], [0, 0]]
    
    # M6 = (c-a)(e+f)
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[0, 0], [0, 1]]
    
    # M7 = (b-d)(g+h)
    U7 = [[0, 1], [0, -1]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[1, 0], [0, 0]]
    
    terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    col = 0
    # c: column block index (0..3)
    # s: row block index of A (0 for cols 0,1; 1 for cols 2,3)
    for c in range(4):
        for s in range(2):
            for t in range(7):
                U_t, V_t, W_t = terms[t]
                
                for u in range(2):
                    for v in range(2):
                        # Determine global j
                        if s == 0:
                            j = v
                        else:
                            j = 2 + v
                        
                        u_val = U_t[u][v]
                        if u_val != 0:
                            u_idx = u * m + j
                            U[u_idx][col] += u_val
                        
                        for w in range(2):
                            k = 2 * c + w
                            
                            v_val = V_t[v][w]
                            if v_val != 0:
                                v_idx = j * p + k
                                V[v_idx][col] += v_val
                            
                            w_val = W_t[u][w]
                            if w_val != 0:
                                w_idx = k * n + u
                                W[w_idx][col] += w_val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
