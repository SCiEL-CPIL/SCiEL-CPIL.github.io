import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_factors(n, m, p):
    """
    Generate factors U, V, W for M_{n,m,p} using recursive Strassen-like decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    # Base case: 2x2x2 (Strassen)
    if n == 2 and m == 2 and p == 2:
        # Strassen algorithm
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map this to U, V, W factors.
        # Each term is a product of linear combinations.
        # Term t: U_t, V_t, W_t
        # U_t has entries for A, V_t for B, W_t for C.
        
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        # A11: i=0,j=0. A22: i=1,j=1.
        # B11: j=0,k=0. B22: j=1,k=1.
        
        # We construct 7 columns.
        U = []
        V = []
        W = []
        
        # Helper to add a term
        def add_term(a_coeffs, b_coeffs, c_coeffs):
            # a_coeffs: list of (i,j, val)
            # b_coeffs: list of (j,k, val)
            # c_coeffs: list of (k,i, val)
            u_col = [0] * (n*m)
            v_col = [0] * (m*p)
            w_col = [0] * (p*n)
            
            for i, j, val in a_coeffs:
                u_col[i*m + j] = val
            for j, k, val in b_coeffs:
                v_col[j*p + k] = val
            for k, i, val in c_coeffs:
                w_col[k*n + i] = val
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            
        # M1
        add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        # M2
        add_term([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]) # C21
        # M3
        add_term([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1)]) # C12
        # M4
        add_term([(1,1,1)], [(0,1,1), (0,0,-1)], [(1,0,1)]) # C11
        # M5
        add_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,-1)]) # C11
        # M6
        add_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]) # C22
        # M7
        add_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]) # C11
        
        return U, V, W

    # Recursive case: Split n into n1, n2
    # We split the dimension that is largest or equal to 2.
    # For simplicity, if n >= 2, split n. Else if m >= 2, split m. Else split p.
    
    # Split n into n1, n2
    if n >= 2:
        n1 = n // 2
        n2 = n - n1
    elif m >= 2:
        m1 = m // 2
        m2 = m - m1
        # Swap roles? No, keep n,m,p fixed.
        # If we split m, we need to handle indices differently.
        # Let's just split n for now. If n=1, split m. If m=1, split p.
        # This code assumes we always split n if possible.
        # If n=1, we must split m or p.
        
    # This logic is flawed for non-cubic.
    # Let's implement a generic split.
    
    # If n > 1, split n.
    if n > 1:
        n1 = n // 2
        n2 = n - n1
        # 8 sub-problems:
        # (n1, m, p), (n2, m, p) etc? No.
        # C = A B.
        # A is n x m. B is m x p.
        # Split A into A1 (n1 x m), A2 (n2 x m).
        # B is not split.
        # C1 = A1 B (n1 x p)
        # C2 = A2 B (n2 x p)
        # This is not Strassen. This is just block multiplication.
        # Rank is R(n1, m, p) + R(n2, m, p).
        
        U1, V1, W1 = _strassen_factors(n1, m, p)
        U2, V2, W2 = _strassen_factors(n2, m, p)
        
        # Combine
        # U rows: 0..n1*m-1 for U1, n1*m..n*m-1 for U2.
        # V rows: same for both.
        # W rows: 0..p*n1-1 for W1, p*n1..p*n-1 for W2.
        
        rank1 = len(U1)
        rank2 = len(U2)
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n*m)]
        V = [[0] * total_rank for _ in range(m*p)]
        W = [[0] * total_rank for _ in range(p*n)]
        
        for c in range(rank1):
            for r in range(n1*m):
                U[r][c] = U1[r][c]
            for r in range(m*p):
                V[r][c] = V1[r][c]
            for r in range(p*n1):
                W[r][c] = W1[r][c]
                
        for c in range(rank2):
            for r in range(n2*m):
                U[n1*m + r][rank1 + c] = U2[r][c]
            for r in range(m*p):
                V[r][rank1 + c] = V2[r][c]
            for r in range(p*n2):
                W[p*n1 + r][rank1 + c] = W2[r][c]
                
        return U, V, W

    elif m > 1:
        m1 = m // 2
        m2 = m - m1
        # Split B into B1 (m1 x p), B2 (m2 x p)? No.
        # A is n x m. B is m x p.
        # Split m.
        # A1 (n x m1), A2 (n x m2).
        # B1 (m1 x p), B2 (m2 x p).
        # C = A1 B1 + A2 B2.
        
        U1, V1, W1 = _strassen_factors(n, m1, p)
        U2, V2, W2 = _strassen_factors(n, m2, p)
        
        rank1 = len(U1)
        rank2 = len(U2)
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n*m)]
        V = [[0] * total_rank for _ in range(m*p)]
        W = [[0] * total_rank for _ in range(p*n)]
        
        for c in range(rank1):
            for r in range(n*m1):
                U[r][c] = U1[r][c]
            for r in range(m1*p):
                V[r][c] = V1[r][c]
            for r in range(p*n):
                W[r][c] = W1[r][c]
                
        for c in range(rank2):
            for r in range(n*m2):
                U[n*m1 + r][rank1 + c] = U2[r][c]
            for r in range(m2*p):
                V[m1*p + r][rank1 + c] = V2[r][c]
            for r in range(p*n):
                W[r][rank1 + c] = W2[r][c]
                
        return U, V, W

    else:
        # p > 1
        p1 = p // 2
        p2 = p - p1
        # Split p.
        # A is n x m. B is m x p.
        # B1 (m x p1), B2 (m x p2).
        # A is same.
        # C1 = A B1 (n x p1), C2 = A B2 (n x p2).
        
        U1, V1, W1 = _strassen_factors(n, m, p1)
        U2, V2, W2 = _strassen_factors(n, m, p2)
        
        rank1 = len(U1)
        rank2 = len(U2)
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n*m)]
        V = [[0] * total_rank for _ in range(m*p)]
        W = [[0] * total_rank for _ in range(p*n)]
        
        for c in range(rank1):
            for r in range(n*m):
                U[r][c] = U1[r][c]
            for r in range(m*p1):
                V[r][c] = V1[r][c]
            for r in range(p1*n):
                W[r][c] = W1[r][c]
                
        for c in range(rank2):
            for r in range(n*m):
                U[r][rank1 + c] = U2[r][c]
            for r in range(m*p2):
                V[m*p1 + r][rank1 + c] = V2[r][c]
            for r in range(p2*n):
                W[p1*n + r][rank1 + c] = W2[r][c]
                
        return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _strassen_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

