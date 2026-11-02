import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    # Strassen for 4 pairs of columns: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        
        # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        # B indices: e=B[0,k0]=k0, f=B[0,k1]=k1, g=B[1,k0]=9+k0, h=B[1,k1]=9+k1
        # C indices: C[0,k0]=2*k0, C[0,k1]=2*k1, C[1,k0]=2*k0+1, C[1,k1]=2*k1+1
        
        # p1 = (a+d)(e+h)
        terms.append(({0:1, 3:1}, {k0:1, 9+k1:1}, {2*k0:1, 2*k1+1:1}))
        # p2 = (c+d)(e)
        terms.append(({2:1, 3:1}, {k0:1}, {2*k0+1:1, 2*k1+1:-1}))
        # p3 = a(f-h)
        terms.append(({0:1}, {k1:1, 9+k1:-1}, {2*k1:1, 2*k1+1:1}))
        # p4 = d(g-e)
        terms.append(({3:1}, {9+k0:1, k0:-1}, {2*k0:1, 2*k0+1:1}))
        # p5 = (a+b)(h)
        terms.append(({0:1, 1:1}, {9+k1:1}, {2*k0:-1, 2*k1:1}))
        # p6 = (c-a)(e+f)
        terms.append(({2:1, 0:-1}, {k0:1, k1:1}, {2*k1+1:1}))
        # p7 = (b-d)(g+h)
        terms.append(({1:1, 3:-1}, {9+k0:1, 9+k1:1}, {2*k0:1}))
    
    # Schoolbook for column 8
    k = 8
    terms.append(({0:1}, {k:1}, {2*k:1}))      # A[0,0]*B[0,8] -> C[0,8]
    terms.append(({1:1}, {9+k:1}, {2*k:1}))     # A[0,1]*B[1,8] -> C[0,8]
    terms.append(({2:1}, {k:1}, {2*k+1:1}))     # A[1,0]*B[0,8] -> C[1,8]
    terms.append(({3:1}, {9+k:1}, {2*k+1:1}))   # A[1,1]*B[1,8] -> C[1,8]
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
