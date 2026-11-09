"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    U shape (4,7), V shape (4,7), W shape (4,7)
    T[i,j,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
    """
    # Strassen's 7 multiplications
    # C = A * B where A, B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i*2+j, r] corresponds to coefficient of A[i,j] in multiplication r
    # V[j*2+k, r] corresponds to coefficient of B[j,k] in multiplication r
    # W[k*2+i, r] corresponds to coefficient of C[i,k] in multiplication r
    
    # Multiplication 1: (A00+A11)(B00+B11)
    # U: A00->1, A11->1; V: B00->1, B11->1; W: C00->1, C11->1
    # Multiplication 2: (A10+A11)B00
    # U: A10->1, A11->1; V: B00->1; W: C10->1, C11->-1
    # Multiplication 3: A00(B01-B11)
    # U: A00->1; V: B01->1, B11->-1; W: C01->1, C11->1
    # Multiplication 4: A11(B10-B00)
    # U: A11->1; V: B10->1, B00->-1; W: C00->1, C10->1
    # Multiplication 5: (A00+A01)B11
    # U: A00->1, A01->1; V: B11->1; W: C00->-1, C01->1
    # Multiplication 6: (A10-A00)(B00+B01)
    # U: A10->1, A00->-1; V: B00->1, B01->1; W: C11->1
    # Multiplication 7: (A01-A11)(B10+B11)
    # U: A01->1, A11->-1; V: B10->1, B11->1; W: C00->1, C01->-1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2: (A10+A11)B00 -> C10+=, C11-=
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1; W[3][1] = -1  # C10, C11
    
    # M3: A00(B01-B11) -> C01+=, C11+=
    U[0][2] = 1  # A00
    V[1][2] = 1; V[3][2] = -1  # B01, B11
    W[1][2] = 1; W[3][2] = 1  # C01, C11
    
    # M4: A11(B10-B00) -> C00+=, C10+=
    U[3][3] = 1  # A11
    V[2][3] = 1; V[0][3] = -1  # B10, B00
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # M5: (A00+A01)B11 -> C00-=, C01+=
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # M6: (A10-A00)(B00+B01) -> C11+=
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11) -> C00+=, C01-=
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1; W[1][6] = -1  # C00, C01
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decomps(n, m, p, U1, V1, W1, U2, V2, W2, split_dim, split_pos):
    """
    Combine two decompositions.
    split_dim: 0 for n-split, 1 for m-split, 2 for p-split
    split_pos: the size of the first part
    """
    if split_dim == 0:  # Split n: n = n1 + n2
        n1 = split_pos
        n2 = n - n1
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        
        new_U = [[0] * r for _ in range(n * m)]
        new_V = [[0] * r for _ in range(m * p)]
        new_W = [[0] * r for _ in range(p * n)]
        
        # U: rows 0..n1*m-1 from U1, rows n1*m..n*m-1 from U2
        for i in range(n1):
            for j in range(m):
                new_U[i * m + j] = U1[i * m + j][:]
        for i in range(n2):
            for j in range(m):
                new_U[(n1 + i) * m + j] = U2[i * m + j][:]
        
        # V: combine element-wise
        for j in range(m):
            for k in range(p):
                new_V[j * p + k] = [v1 + v2 for v1, v2 in zip(V1[j * p + k], V2[j * p + k])]
        
        # W: rows 0..p*n1-1 from W1, rows p*n1..p*n-1 from W2
        for k in range(p):
            for i in range(n1):
                new_W[k * n + i] = W1[k * n1 + i][:]
            for i in range(n2):
                new_W[k * n + (n1 + i)] = W2[k * n2 + i][:]
        
        return new_U, new_V, new_W
    
    elif split_dim == 1:  # Split m: m = m1 + m2
        m1 = split_pos
        m2 = m - m1
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        
        new_U = [[0] * r for _ in range(n * m)]
        new_V = [[0] * r for _ in range(m * p)]
        new_W = [[0] * r for _ in range(p * n)]
        
        # U: for each i, first m1 columns from U1, next m2 from U2
        for i in range(n):
            for j in range(m1):
                new_U[i * m + j] = U1[i * m1 + j][:]
            for j in range(m2):
                new_U[i * m + (m1 + j)] = U2[i * m2 + j][:]
        
        # V: rows 0..m1*p-1 from V1, rows m1*p..m*p-1 from V2
        for j in range(m1):
            for k in range(p):
                new_V[j * p + k] = V1[j * p + k][:]
        for j in range(m2):
            for k in range(p):
                new_V[(m1 + j) * p + k] = V2[j * p + k][:]
        
        # W: combine element-wise
        for k in range(p):
            for i in range(n):
                new_W[k * n + i] = [w1 + w2 for w1, w2 in zip(W1[k * n + i], W2[k * n + i])]
        
        return new_U, new_V, new_W
    
    else:  # Split p: p = p1 + p2
        p1 = split_pos
        p2 = p - p1
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        
        new_U = [[0] * r for _ in range(n * m)]
        new_V = [[0] * r for _ in range(m * p)]
        new_W = [[0] * r for _ in range(p * n)]
        
        # U: combine element-wise
        for i in range(n):
            for j in range(m):
                new_U[i * m + j] = [u1 + u2 for u1, u2 in zip(U1[i * m + j], U2[i * m + j])]
        
        # V: for each j, first p1 columns from V1, next p2 from V2
        for j in range(m):
            for k in range(p1):
                new_V[j * p + k] = V1[j * p1 + k][:]
            for k in range(p2):
                new_V[j * p + (p1 + k)] = V2[j * p2 + k][:]
        
        # W: rows 0..p1*n-1 from W1, rows p1*n..p*n-1 from W2
        for k in range(p1):
            for i in range(n):
                new_W[k * n + i] = W1[k * n + i][:]
        for k in range(p2):
            for i in range(n):
                new_W[(p1 + k) * n + i] = W2[k * n + i][:]
        
        return new_U, new_V, new_W

def _get_decomp(n, m, p, deadline):
    """
    Get a decomposition for n x m * m x p.
    Returns (U, V, W, rank) where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    # Base cases
    if n == 1 and m == 1 and p == 1:
        U, V, W = _schoolbook_nm_p(1, 1, 1)
        return U, V, W, 1
    
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    
    # Check deadline
    if time.time() > deadline - 5:
        U, V, W = _schoolbook_nm_p(n, m, p)
        return U, V, W, n * m * p
    
    # Try all splits and pick the one with minimum rank
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_nm_p(n, m, p)
    
    # Try splitting n
    if n > 1:
        for n1 in range(1, n):
            if time.time() > deadline - 5:
                break
            U1, V1, W1, r1 = _get_decomp(n1, m, p, deadline)
            U2, V2, W2, r2 = _get_decomp(n - n1, m, p, deadline)
            r = r1 + r2
            if r < best_rank:
                new_U, new_V, new_W = _combine_decomps(n, m, p, U1, V1, W1, U2, V2, W2, 0, n1)
                best_rank = r
                best_U, best_V, best_W = new_U, new_V, new_W
    
    # Try splitting m
    if m > 1:
        for m1 in range(1, m):
            if time.time() > deadline - 5:
                break
            U1, V1, W1, r1 = _get_decomp(n, m1, p, deadline)
            U2, V2, W2, r2 = _get_decomp(n, m - m1, p, deadline)
            r = r1 + r2
            if r < best_rank:
                new_U, new_V, new_W = _combine_decomps(n, m, p, U1, V1, W1, U2, V2, W2, 1, m1)
                best_rank = r
                best_U, best_V, best_W = new_U, new_V, new_W
    
    # Try splitting p
    if p > 1:
        for p1 in range(1, p):
            if time.time() > deadline - 5:
                break
            U1, V1, W1, r1 = _get_decomp(n, m, p1, deadline)
            U2, V2, W2, r2 = _get_decomp(n, m, p - p1, deadline)
            r = r1 + r2
            if r < best_rank:
                new_U, new_V, new_W = _combine_decomps(n, m, p, U1, V1, W1, U2, V2, W2, 2, p1)
                best_rank = r
                best_U, best_V, best_W = new_U, new_V, new_W
    
    return best_U, best_V, best_W, best_rank

def _verify_decomp(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    U, V, W, rank = _get_decomp(n, m, p, deadline)
    
    # Verify
    if not _verify_decomp(n, m, p, U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook_nm_p(n, m, p)
    
    return {"U": U, "V": V, "W": W}
