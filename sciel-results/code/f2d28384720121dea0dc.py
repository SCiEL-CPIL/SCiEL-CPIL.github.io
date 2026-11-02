# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with Strassen rank 7.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2*2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    W index: k*n+i where k in {0,1}, i in {0,1}
    U index: i*m+j where i in {0,1}, j in {0,1}
    V index: j*p+k where j in {0,1}, k in {0,1}
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Standard Strassen:
    # C11 = P1 - P4 + P5 - P7
    # C12 = P3 + P7
    # C21 = P2 + P6
    # C22 = P1 - P3 + P4 + P6
    
    # U rows: a11=0, a12=1, a21=2, a22=3
    # V rows: b11=0, b12=1, b21=2, b22=3
    # W rows: c11=0, c12=1, c21=2, c22=3
    
    # Column 0: P1 = (a11+a22)(b11+b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # Column 1: P2 = (a21+a22)b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 1]
    
    # Column 2: P3 = a11(b12-b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, -1]
    
    # Column 3: P4 = a22(b21-b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [-1, 0, 0, 1]
    
    # Column 4: P5 = (a11+a12)b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [1, 0, 0, 1]
    
    # Column 5: P6 = (a21-a11)(b11+b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 1, 1]
    
    # Column 6: P7 = (a12-a22)(b21+b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [0, 1, 0, -1]
    
    return U, V, W

def _identity_block(n, m, p):
    """Return trivial rank n*m*p decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _decompose_nm_p(n, m, p):
    """Decompose n x m * m x p matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2 blocks.
    Returns U, V, W as numpy arrays.
    """
    # Base case: if any dimension is 0, return empty
    if n == 0 or m == 0 or p == 0:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W
    
    # Base case: 2x2*2x2 -> Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Base case: if m == 1, it's just outer product scaled
    if m == 1:
        rank = n * p
        U = np.zeros((n*1, rank))
        V = np.zeros((1*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[0, col] = 1
                W[k*n + i, col] = 1
                col += 1
        return U, V, W
    
    # If n == 1
    if n == 1:
        rank = m * p
        U = np.zeros((1*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*1, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j*p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
    
    # If p == 1
    if p == 1:
        rank = n * m
        U = np.zeros((n*m, rank))
        V = np.zeros((m*1, rank))
        W = np.zeros((1*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def try_split(dim, split_idx, n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
        nonlocal best_rank, best_U, best_V, best_W
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            if dim == 'm':
                # U: rows i*m+j. j in [0, m1) for U1, j in [m1, m) for U2
                for i in range(n):
                    for j in range(m1):
                        U[i*m + j, :r1] = U1[i*m1 + j, :]
                    for j in range(m2):
                        U[i*m + m1 + j, r1:] = U2[i*m2 + j, :]
                
                # V: rows j*p+k. j in [0, m1) for V1, j in [m1, m) for V2
                V[:m1*p, :r1] = V1
                V[m1*p:, r1:] = V2
                
                # W: rows k*n+i. Same for both
                W[:, :r1] = W1
                W[:, r1:] = W2
                
            elif dim == 'n':
                # U: rows i*m+j. i in [0, n1) for U1, i in [n1, n) for U2
                for i in range(n1):
                    for j in range(m):
                        U[i*m + j, :r1] = U1[i*m1 + j, :]
                for i in range(n2):
                    for j in range(m):
                        U[(n1+i)*m + j, r1:] = U2[i*m2 + j, :]
                
                # V: rows j*p+k. Same for both
                V[:, :r1] = V1
                V[:, r1:] = V2
                
                # W: rows k*n+i. i in [0, n1) for W1, i in [n1, n) for W2
                for k in range(p):
                    for i in range(n1):
                        W[k*n + i, :r1] = W1[k*n1 + i, :]
                    for i in range(n2):
                        W[k*n + n1 + i, r1:] = W2[k*n2 + i, :]
                        
            elif dim == 'p':
                # U: rows i*m+j. Same for both
                U[:, :r1] = U1
                U[:, r1:] = U2
                
                # V: rows j*p+k. k in [0, p1) for V1, k in [p1, p) for V2
                for j in range(m):
                    for k in range(p1):
                        V[j*p + k, :r1] = V1[j*p1 + k, :]
                    for k in range(p2):
                        V[j*p + p1 + k, r1:] = V2[j*p2 + k, :]
                
                # W: rows k*n+i. k in [0, p1) for W1, k in [p1, p) for W2
                for k in range(p1):
                    for i in range(n):
                        W[k*n + i, :r1] = W1[k*n1 + i, :]
                for k in range(p2):
                    for i in range(n):
                        W[(p1+k)*n + i, r1:] = W2[k*n2 + i, :]
                        
            best_U = U
            best_V = V
            best_W = W

    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0: continue
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        try_split('m', m1, n, m1, p, n, m2, p, U1, V1, W1, U2, V2, W2)

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        if n2 <= 0: continue
        U1, V1, W1 = _decompose_nm_p(n1, m, p)
        U2, V2, W2 = _decompose_nm_p(n2, m, p)
        try_split('n', n1, n1, m, p, n2, m, p, U1, V1, W1, U2, V2, W2)

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        if p2 <= 0: continue
        U1, V1, W1 = _decompose_nm_p(n, m, p1)
        U2, V2, W2 = _decompose_nm_p(n, m, p2)
        try_split('p', p1, n, m, p1, n, m, p2, U1, V1, W1, U2, V2, W2)

    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback: schoolbook
    return _identity_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose_nm_p(n, m, p)
    rank = U.shape[1]
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in U.tolist()]
    V_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in V.tolist()]
    W_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
