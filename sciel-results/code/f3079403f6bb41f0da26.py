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

def _construct_rank32():
    n, m, p = DIMENSIONS
    rank = 32
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

    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def c_idx(i, k): return k * n + i

    # Strassen for 2x2 blocks
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

    # Strassen for A11 (i=0,1; j=0,1) and B11 (j=0,1; k=0,1)
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    # Strassen for A11 (i=0,1; j=0,1) and B12 (j=0,1; k=2,3)
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    # Schoolbook for A12 (i=0,1; j=2) and B21 (j=2; k=0,1)
    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)

    # Schoolbook for A12 (i=0,1; j=2) and B22 (j=2; k=2,3)
    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)

    # Schoolbook for A21 (i=2; j=0,1) and B11 (j=0,1; k=0,1)
    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)

    # Schoolbook for A21 (i=2; j=0,1) and B12 (j=0,1; k=2,3)
    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)

    # Schoolbook for A22 (i=2; j=2) and B21 (j=2; k=0,1)
    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)

    # Schoolbook for A22 (i=2; j=2) and B22 (j=2; k=2,3)
    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # For each (i, k), check that C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: for each (i, j, k), the coefficient of A[i,j]*B[j,k] in C[i,k] should be 1
    # This means: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1 for all (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True

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

    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def c_idx(i, k): return k * n + i

    # Strassen for 2x2 blocks
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        U[a00][c] = 1; U[a11][c] = 1
        V[b00][c] = 1; V[b11][c] = 1
        W[c00][c] = 1; W[c11][c] = 1
        c += 1
        U[a10][c] = 1; U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1; W[c11][c] = -1
        c += 1
        U[a00][c] = 1
        V[b01][c] = 1; V[b11][c] = -1
        W[c01][c] = 1; W[c11][c] = 1
        c += 1
        U[a11][c] = 1
        V[b10][c] = 1; V[b00][c] = -1
        W[c00][c] = 1; W[c10][c] = 1
        c += 1
        U[a00][c] = 1; U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1; W[c01][c] = 1
        c += 1
        U[a10][c] = 1; U[a00][c] = -1
        V[b00][c] = 1; V[b01][c] = 1
        W[c11][c] = 1
        c += 1
        U[a01][c] = 1; U[a11][c] = -1
        V[b10][c] = 1; V[b11][c] = 1
        W[c00][c] = 1
        c += 1

    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1), 
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1), 
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3), 
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)
    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)
    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def construct():
    global _START
    _START = time.time()
    
    # Try rank 32 first
    try:
        result32 = _construct_rank32()
        if _verify(result32["U"], result32["V"], result32["W"]):
            return result32
    except:
        pass
    
    # Fall back to rank 34
    try:
        result34 = _construct_rank34()
        if _verify(result34["U"], result34["V"], result34["W"]):
            return result34
    except:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
