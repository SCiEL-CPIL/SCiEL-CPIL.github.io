import time

DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    # Helper to add Strassen terms for 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        # M1 = (a00+a11)(b00+b11)
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        # M2 = (a10+a11)b00
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        # M3 = a00(b01-b11)
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        # M4 = a11(b10-b00)
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        # M5 = (a00+a01)b11
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        # M6 = (a10-a00)(b00+b01)
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        # M7 = (a01-a11)(b10+b11)
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    # Indices
    # A: i*3+j
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    a20, a21, a22 = 6, 7, 8
    
    # B: j*4+k
    b00, b01, b02, b03 = 0, 1, 2, 3
    b10, b11, b12, b13 = 4, 5, 6, 7
    b20, b21, b22, b23 = 8, 9, 10, 11
    
    # C: k*3+i
    c00, c01, c02, c03 = 0, 1, 2, 3
    c10, c11, c12, c13 = 3, 4, 5, 6
    c20, c21, c22, c23 = 6, 7, 8, 9
    # Wait, C index is k*n+i. n=3.
    # k=0: 0, 1, 2 (i=0,1,2) -> c00, c01, c02
    # k=1: 3, 4, 5 (i=0,1,2) -> c10, c11, c12
    # k=2: 6, 7, 8 (i=0,1,2) -> c20, c21, c22
    # k=3: 9, 10, 11 (i=0,1,2) -> c30, c31, c32
    
    # Correction:
    # C[i,k] is at index k*3+i.
    # C00 (i=0,k=0) -> 0
    # C01 (i=0,k=1) -> 3
    # C02 (i=0,k=2) -> 6
    # C03 (i=0,k=3) -> 9
    # C10 (i=1,k=0) -> 1
    # C11 (i=1,k=1) -> 4
    # C12 (i=1,k=2) -> 7
    # C13 (i=1,k=3) -> 10
    # C20 (i=2,k=0) -> 2
    # C21 (i=2,k=1) -> 5
    # C22 (i=2,k=2) -> 8
    # C23 (i=2,k=3) -> 11

    # Map block C indices to global C indices
    # Block 11 (i=0,1; k=0,1)
    c11_00 = c00 # 0
    c11_01 = c10 # 3? No. C01 is i=0,k=1. Index 1*3+0=3.
    # My previous mapping was wrong.
    
    # Let's use a helper to get index
    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    # Strassen for A11 (i=0,1; j=0,1) and B11 (j=0,1; k=0,1)
    # A indices: 0,1; 3,4
    # B indices: 0,1; 4,5
    # C indices: c(0,0)=0, c(0,1)=3, c(1,0)=1, c(1,1)=4
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    # Strassen for A11 (i=0,1; j=0,1) and B12 (j=0,1; k=2,3)
    # B indices: v(0,2)=2, v(0,3)=3; v(1,2)=6, v(1,3)=7
    # C indices: c(0,2)=6, c(0,3)=9, c(1,2)=7, c(1,3)=10
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    # Schoolbook for A12 (i=0,1; j=2) and B21 (j=2; k=0,1)
    # A: u(0,2)=2, u(1,2)=5
    # B: v(2,0)=8, v(2,1)=9
    # C: c(0,0)=0, c(0,1)=3, c(1,0)=1, c(1,1)=4
    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)

    # Schoolbook for A12 (i=0,1; j=2) and B22 (j=2; k=2,3)
    # B: v(2,2)=10, v(2,3)=11
    # C: c(0,2)=6, c(0,3)=9, c(1,2)=7, c(1,3)=10
    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)

    # Schoolbook for A21 (i=2; j=0,1) and B11 (j=0,1; k=0,1)
    # A: u(2,0)=6, u(2,1)=7
    # B: v(0,0)=0, v(0,1)=1, v(1,0)=4, v(1,1)=5
    # C: c(2,0)=2, c(2,1)=5
    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)

    # Schoolbook for A21 (i=2; j=0,1) and B12 (j=0,1; k=2,3)
    # B: v(0,2)=2, v(0,3)=3, v(1,2)=6, v(1,3)=7
    # C: c(2,2)=8, c(2,3)=11
    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)

    # Schoolbook for A22 (i=2; j=2) and B21 (j=2; k=0,1)
    # A: u(2,2)=8
    # B: v(2,0)=8, v(2,1)=9
    # C: c(2,0)=2, c(2,1)=5
    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)

    # Schoolbook for A22 (i=2; j=2) and B22 (j=2; k=2,3)
    # B: v(2,2)=10, v(2,3)=11
    # C: c(2,2)=8, c(2,3)=11
    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def construct():
    global _START
    _START = time.time()
    try:
        return _construct_rank34()
    except:
        return _schoolbook()
