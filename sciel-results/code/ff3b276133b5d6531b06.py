<<<<<<< SEARCH
    # Try to split m into m1 + m2 where m1, m2 > 0
    # Choose split that minimizes total rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        # Get decompositions for (n, m1, p) and (n, m2, p)
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            # Combine: U1 and U2 have different row indices because m1 != m2
            # U1 has n*m1 rows, U2 has n*m2 rows
            # We need to interleave them properly.
            # Actually, the rows of U correspond to (i, j) where i in [0,n), j in [0,m)
            # For m1: j in [0, m1)
            # For m2: j in [m1, m)
            # So U is block diagonal with U1 (n*m1 rows) and U2 (n*m2 rows)
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # U1 goes in rows 0..n*m1-1, columns 0..r1-1
            # But we need to map the j indices correctly.
            # U1[i*m1 + j, c] corresponds to A[i, j] where j in [0, m1)
            # In the full U, A[i, j] is at row i*m + j
            # So for j in [0, m1): full_row = i*m + j
            # U1 row index: i*m1 + j
            # So full_row = i*m + j = (i*m1 + j) + i*(m - m1) = U1_row + i*(m-m1)
            # This is not a simple block diagonal; we need to scatter the rows.
            
            for i in range(n):
                for j in range(m1):
                    u1_row = i * m1 + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            for i in range(n):
                for j in range(m2):
                    u2_row = i * m2 + j
                    full_row = i * m + (m1 + j)
                    U[full_row, r1:] = U2[u2_row, :]
            
            # V1: rows are (j, k) where j in [0, m1), k in [0, p)
            # V1[j*p + k, c]
            # Full V: (j, k) where j in [0, m), k in [0, p)
            # Full row: j*p + k
            # V1 row: j*p + k for j in [0, m1)
            # So V1 rows 0..m1*p-1 map directly to full V rows 0..m1*p-1
            V[:m1*p, :r1] = V1
            # V2: j in [0, m2), full j' = m1 + j
            # Full row: (m1+j)*p + k = m1*p + j*p + k
            # V2 row: j*p + k
            V[m1*p:, r1:] = V2
            
            # W is the same for both since the output dimensions are the same
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U = U
            best_V = V
            best_W = W
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback: schoolbook
    return _identity_block(n, m, p)
=======
    # Try splitting along m dimension
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            for i in range(n):
                for j in range(m1):
                    U[i*m + j, :r1] = U1[i*m1 + j, :]
            for i in range(n):
                for j in range(m2):
                    U[i*m + (m1+j), r1:] = U2[i*m2 + j, :]
            
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U = U
            best_V = V
            best_W = W
    
    # Try splitting along n dimension (output rows)
    for n1 in range(1, n):
        n2 = n - n1
        if n2 <= 0:
            continue
        U1, V1, W1 = _decompose_nm_p(n1, m, p)
        U2, V2, W2 = _decompose_nm_p(n2, m, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # U: rows are (i,j) where i in [0,n), j in [0,m)
            # For n1: i in [0,n1), full row = i*m + j
            # U1 row: i*m + j for i in [0,n1)
            U[:n1*m, :r1] = U1
            # For n2: i in [0,n2), full i' = n1 + i
            # Full row: (n1+i)*m + j = n1*m + i*m + j
            U[n1*m:, r1:] = U2
            
            # V is the same for both
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            # W: rows are (k,i) where k in [0,p), i in [0,n)
            # W row index: k*n + i
            # For n1: i in [0,n1)
            # For n2: i in [0,n2), full i' = n1 + i
            # This is tricky because W is indexed by k*n+i, not block-diagonal in n
            for k in range(p):
                for i in range(n1):
                    W[k*n + i, :r1] = W1[k*n1 + i, :]
                for i in range(n2):
                    W[k*n + (n1+i), r1:] = W2[k*n2 + i, :]
            
            best_U = U
            best_V = V
            best_W = W
    
    # Try splitting along p dimension (output columns)
    for p1 in range(1, p):
        p2 = p - p1
        if p2 <= 0:
            continue
        U1, V1, W1 = _decompose_nm_p(n, m, p1)
        U2, V2, W2 = _decompose_nm_p(n, m, p2)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # U is the same for both
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # V: rows are (j,k) where j in [0,m), k in [0,p)
            # V row index: j*p + k
            # For p1: k in [0,p1)
            # For p2: k in [0,p2), full k' = p1 + k
            for j in range(m):
                for k in range(p1):
                    V[j*p + k, :r1] = V1[j*p1 + k, :]
                for k in range(p2):
                    V[j*p + (p1+k), r1:] = V2[j*p2 + k, :]
            
            # W: rows are (k,i) where k in [0,p), i in [0,n)
            # W row index: k*n + i
            # For p1: k in [0,p1)
            # For p2: k in [0,p2), full k' = p1 + k
            for k in range(p1):
                for i in range(n):
                    W[k*n + i, :r1] = W1[k*p1_n + i, :] if False else W1[k*n + i, :]
            # Wait, W1 has p1*n rows, indexed by k*n + i where k in [0,p1)
            # But in W1, the index is k*n + i with n being the same n
            # Actually W1 shape is (p1*n, r1), indexed by k*n + i
            # In full W, index is k*n + i
            # So for k in [0,p1): W[k*n + i] = W1[k*n + i]
            # For k in [0,p2): full_k = p1 + k, W[(p1+k)*n + i] = W2[k*n + i]
            for k in range(p1):
                for i in range(n):
                    W[k*n + i, :r1] = W1[k*n + i, :]
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n + i, r1:] = W2[k*n + i, :]
            
            best_U = U
            best_V = V
            best_W = W
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback: schoolbook
    return _identity_block(n, m, p)
>>>>>>> REPLACE

Wait, I made an error in the p-splitting section. Let me fix that:

