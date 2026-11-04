import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(u_idx, v_idx, w_idx, r_start):
        # u_idx: [a00, a01, a10, a11]
        # v_idx: [b00, b01, b10, b11]
        # w_idx: [c00, c01, c10, c11]
        # Note: w_idx order corresponds to c00, c01, c10, c11
        # But W index is k*n+i.
        # c00 (i=0,k=0) -> w_idx[0]
        # c01 (i=0,k=1) -> w_idx[1]
        # c10 (i=1,k=0) -> w_idx[2]
        # c11 (i=1,k=1) -> w_idx[3]
        
        a00, a01, a10, a11 = u_idx
        b00, b01, b10, b11 = v_idx
        c00, c01, c10, c11 = w_idx
        
        # M1 = (a00 + a11)(b00 + b11)
        r = r_start
        U[a00][r] += 1; U[a11][r] += 1
        V[b00][r] += 1; V[b11][r] += 1
        W[c00][r] += 1
        W[c11][r] += 1
        r += 1
        
        # M2 = (a10 + a11)b00
        U[a10][r] += 1; U[a11][r] += 1
        V[b00][r] += 1
        W[c10][r] += 1
        W[c11][r] -= 1
        r += 1
        
        # M3 = a00(b01 - b11)
        U[a00][r] += 1
        V[b01][r] += 1; V[b11][r] -= 1
        W[c01][r] += 1
        W[c11][r] += 1
        r += 1
        
        # M4 = a11(b10 - b00)
        U[a11][r] += 1
        V[b10][r] += 1; V[b00][r] -= 1
        W[c00][r] += 1
        W[c10][r] += 1
        r += 1
        
        # M5 = (a00 + a01)b11
        U[a00][r] += 1; U[a01][r] += 1
        V[b11][r] += 1
        W[c00][r] -= 1
        W[c01][r] += 1
        r += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a10][r] += 1; U[a00][r] -= 1
        V[b00][r] += 1; V[b01][r] += 1
        W[c11][r] += 1
        r += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a01][r] += 1; U[a11][r] -= 1
        V[b10][r] += 1; V[b11][r] += 1
        W[c00][r] += 1
        r += 1
        
        return r_start + 7

    def add_schoolbook_2x2x1(u_idx, v_idx, w_idx, r_start):
        # u_idx: [a00, a01, a10, a11]
        # v_idx: [b00, b10] (B is 2x1)
        # w_idx: [c00, c10] (C is 2x1)
        
        a00, a01, a10, a11 = u_idx
        b00, b10 = v_idx
        c00, c10 = w_idx
        
        # c00 = a00*b00 + a01*b10
        r = r_start
        U[a00][r] += 1
        V[b00][r] += 1
        W[c00][r] += 1
        r += 1
        
        U[a01][r] += 1
        V[b10][r] += 1
        W[c00][r] += 1
        r += 1
        
        # c10 = a10*b00 + a11*b10
        U[a10][r] += 1
        V[b00][r] += 1
        W[c10][r] += 1
        r += 1
        
        U[a11][r] += 1
        V[b10][r] += 1
        W[c10][r] += 1
        r += 1
        
        return r_start + 4

    # Indices
    # A: 2x4. Row 0: 0,1,2,3. Row 1: 4,5,6,7.
    # B: 4x5. Row 0: 0-4. Row 1: 5-9. Row 2: 10-14. Row 3: 15-19.
    # C: 2x5. W index k*2+i.
    # k=0: 0,1. k=1: 2,3. k=2: 4,5. k=3: 6,7. k=4: 8,9.
    
    # Block 1: A_L (cols 0,1) x B_T1 (rows 0,1, cols 0,1)
    # A_L: a00=0, a01=1, a10=4, a11=5
    # B_T1: b00=0, b01=1, b10=5, b11=6
    # C: k=0,1. i=0,1.
    # c00 (i=0,k=0) -> W[0]
    # c01 (i=0,k=1) -> W[2]
    # c10 (i=1,k=0) -> W[1]
    # c11 (i=1,k=1) -> W[3]
    r = add_strassen([0, 1, 4, 5], [0, 1, 5, 6], [0, 2, 1, 3], r)
    
    # Block 2: A_L x B_T2 (rows 0,1, cols 2,3)
    # B_T2: b00=2, b01=3, b10=7, b11=8
    # C: k=2,3.
    # c00 (i=0,k=2) -> W[4]
    # c01 (i=0,k=3) -> W[6]
    # c10 (i=1,k=2) -> W[5]
    # c11 (i=1,k=3) -> W[7]
    r = add_strassen([0, 1, 4, 5], [2, 3, 7, 8], [4, 6, 5, 7], r)
    
    # Block 3: A_L x B_T3 (rows 0,1, col 4)
    # B_T3: b00=4, b10=9
    # C: k=4.
    # c00 (i=0,k=4) -> W[8]
    # c10 (i=1,k=4) -> W[9]
    r = add_schoolbook_2x2x1([0, 1, 4, 5], [4, 9], [8, 9], r)
    
    # Block 4: A_R (cols 2,3) x B_B1 (rows 2,3, cols 0,1)
    # A_R: a00=2, a01=3, a10=6, a11=7
    # B_B1: b00=10, b01=11, b10=15, b11=16
    # C: k=0,1.
    # c00 (i=0,k=0) -> W[0]
    # c01 (i=0,k=1) -> W[2]
    # c10 (i=1,k=0) -> W[1]
    # c11 (i=1,k=1) -> W[3]
    r = add_strassen([2, 3, 6, 7], [10, 11, 15, 16], [0, 2, 1, 3], r)
    
    # Block 5: A_R x B_B2 (rows 2,3, cols 2,3)
    # B_B2: b00=12, b01=13, b10=17, b11=18
    # C: k=2,3.
    # c00 (i=0,k=2) -> W[4]
    # c01 (i=0,k=3) -> W[6]
    # c10 (i=1,k=2) -> W[5]
    # c11 (i=1,k=3) -> W[7]
    r = add_strassen([2, 3, 6, 7], [12, 13, 17, 18], [4, 6, 5, 7], r)
    
    # Block 6: A_R x B_B3 (rows 2,3, col 4)
    # B_B3: b00=14, b10=19
    # C: k=4.
    # c00 (i=0,k=4) -> W[8]
    # c10 (i=1,k=4) -> W[9]
    r = add_schoolbook_2x2x1([2, 3, 6, 7], [14, 19], [8, 9], r)
    
    if time.time() > _deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
