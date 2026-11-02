import time

def construct():
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s kill
    
    n, m, p = 2, 2, 12
    rank = 42
    U = [[0]*rank for _ in range(n*m)]  # 4 x 42
    V = [[0]*rank for _ in range(m*p)]  # 24 x 42
    W = [[0]*rank for _ in range(p*n)]  # 24 x 42
    
    num_pairs = p // 2  # 6
    # Strassen products for 2x2 x 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    for t in range(num_pairs):
        # W indices for block t: C(i,k) -> W[k*n+i]
        # C(0,2t) -> W[4t], C(1,2t) -> W[4t+1], C(0,2t+1) -> W[4t+2], C(1,2t+1) -> W[4t+3]
        w_c00 = 4*t
        w_c10 = 4*t+1
        w_c01 = 4*t+2
        w_c11 = 4*t+3
        # B indices for block t: B(j,k) -> V[j*p+k]
        v_b00 = 2*t
        v_b01 = 2*t+1
        v_b10 = 12 + 2*t
        v_b11 = 12 + 2*t + 1
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[2][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] = 1
        V[v_b01][col] = 1
        V[v_b11][col] = -1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[3][col] = 1
        V[v_b10][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] = 1
        U[1][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[v_b00][col] = 1
        V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[v_b10][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
    
    # Time guard: return immediately (construction is O(1))
    if time.time() - start_time > deadline:
        pass  # Should never happen for this O(1) construction
    
    return {"U": U, "V": V, "W": W}
