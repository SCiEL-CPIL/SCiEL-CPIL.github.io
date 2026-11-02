import numpy as np

def construct():
    n, m, p = 3, 5, 7
    U, V, W = decompose(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def decompose(n, m, p):
    # Base case: Schoolbook
    if n == 1 or m == 1 or p == 1:
        return schoolbook(n, m, p)
    
    # Base case: Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        return strassen()
    
    # Recursive case: Split the largest dimension
    # Heuristic: Split into 2 and (dim-2) to maximize Strassen usage
    max_dim = max(n, m, p)
    if max_dim == 2:
        # If all are 2, handled above. If some are >2, split them.
        pass
        
    if n >= 3:
        n1, n2 = 2, n - 2
        U1, V1, W1 = decompose(n1, m, p)
        U2, V2, W2 = decompose(n2, m, p)
        # Concatenate
        # U is (n*m, rank). U1 is (n1*m, r1), U2 is (n2*m, r2)
        # Need to align indices.
        # This simple concatenation doesn't work directly because U, V, W are linked by the term index.
        # We need to construct the full U, V, W for the sum of two products.
        # Product 1: A1 B1. Product 2: A2 B2.
        # Total C = A1 B1 + A2 B2.
        # U_total = [U1, U2] (block diag? No, U maps A to terms. V maps B to terms. W maps terms to C).
        # For sum of two decompositions:
        # U_total = [U1, 0; 0, U2] ? No.
        # A is split into A1, A2.
        # U1 acts on A1. U2 acts on A2.
        # So U_total should be block diagonal?
        # U_total[i, k] = 1 if term k uses A1...
        # Actually, the rank is additive.
        # U_total = np.vstack([U1, U2])? No, dimensions don't match.
        # U1 is (n1*m, r1). U2 is (n2*m, r2).
        # The full U is (n*m, r1+r2).
        # Rows 0..n1*m-1 correspond to A1. Rows n1*m..n*m-1 correspond to A2.
        # U_total[0:n1*m, 0:r1] = U1
        # U_total[n1*m:n*m, r1:r1+r2] = U2
        # Similarly for V and W.
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        U[:n1 * m, :r1] = U1
        V[:m * p, :r1] = V1 # V1 is (m*p, r1)
        W[:p * n1, :r1] = W1 # W1 is (p*n1, r1) -> Wait, W shape is (p*n, rank).
        
        # W1 corresponds to output C1 (n1 x p).
        # W1 is (p*n1, r1).
        # W_total is (p*n, rank).
        # W_total[0:p*n1, 0:r1] = W1
        # W_total[p*n1:p*n, r1:rank] = W2
        
        U[n1 * m:, r1:] = U2
        V[:, r1:] = V2
        W[p * n1:, r1:] = W2
        
        return U, V, W

    elif m >= 3:
        m1, m2 = 2, m - 2
        U1, V1, W1 = decompose(n, m1, p)
        U2, V2, W2 = decompose(n, m2, p)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # U1: (n*m1, r1). U2: (n*m2, r2).
        # U rows: 0..n*m1-1 for A[:, :m1]. n*m1..n*m-1 for A[:, m1:].
        U[:n * m1, :r1] = U1
        U[n * m1:, r1:] = U2
        
        # V1: (m1*p, r1). V2: (m2*p, r2).
        # V rows: 0..m1*p-1 for B[:, :p] (first m1 cols of B? No, B is m x p).
        # V maps B to terms.
        # B is split into B1 (m1 x p) and B2 (m2 x p).
        # V1 acts on B1. V2 acts on B2.
        V[:m1 * p, :r1] = V1
        V[m1 * p:, r1:] = V2
        
        # W1: (p*n, r1). W2: (p*n, r2).
        # W acts on C (n x p).
        # C1 (n x p) and C2 (n x p) are added.
        # W_total[:, 0:r1] = W1
        # W_total[:, r1:rank] = W2
        W[:, :r1] = W1
        W[:, r1:] = W2
        
        return U, V, W

    elif p >= 3:
        p1, p2 = 2, p - 2
        U1, V1, W1 = decompose(n, m, p1)
        U2, V2, W2 = decompose(n, m, p2)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # U1: (n*m, r1). U2: (n*m, r2).
        # U is same for both parts?
        # A is same. So U_total[:, 0:r1] = U1, U_total[:, r1:rank] = U2.
        U[:, :r1] = U1
        U[:, r1:] = U2
        
        # V1: (m*p1, r1). V2: (m*p2, r2).
        # B is split into B1 (m x p1) and B2 (m x p2).
        # V rows: 0..m*p1-1 for B[:, :p1]. m*p1..m*p-1 for B[:, p1:].
        V[:m * p1, :r1] = V1
        V[m * p1:, r1:] = V2
        
        # W1: (p1*n, r1). W2: (p2*n, r2).
        # C is split into C1 (n x p1) and C2 (n x p2).
        # W rows: 0..p1*n-1 for C[:, :p1]. p1*n..p*n-1 for C[:, p1:].
        W[:p1 * n, :r1] = W1
        W[p1 * n:, r1:] = W2
        
        return U, V, W

    else:
        # Should not reach here if logic is correct
        return schoolbook(n, m, p)

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return U, V, W

def strassen():
    # 2x2x2 Strassen decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U (2x2 x 7) -> (4, 7)
    # V (2x2 x 7) -> (4, 7)
    # W (2x2 x 7) -> (4, 7)
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=ae+bg, 1=af+bh, 2=ce+dg, 3=cf+dh
    
    # M1: (a+d)(e+h)
    U[0, 0] = 1; U[0, 3] = 1
    V[0, 0] = 1; V[0, 3] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: (c+d)e
    U[1, 2] = 1; U[1, 3] = 1
    V[1, 0] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: a(f-h)
    U[2, 0] = 1
    V[2, 1] = 1; V[2, 3] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: d(g-e)
    U[3, 3] = 1
    V[3, 2] = 1; V[3, 0] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5: (a+b)h
    U[4, 0] = 1; U[4, 1] = 1
    V[4, 3] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6: (c-a)(e+f)
    U[5, 2] = 1; U[5, 0] = -1
    V[5, 0] = 1; V[5, 1] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7: (b-d)(g+h)
    U[6, 1] = 1; U[6, 3] = -1
    V[6, 2] = 1; V[6, 3] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W
