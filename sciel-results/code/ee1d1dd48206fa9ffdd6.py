def combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Combine M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}."""
    p = p1 + p2
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # U: same for both (same A)
    U[:, :r1] = U1
    U[:, r1:] = U2
    
    # V: first r1 cols in B1 part (cols 0..p1-1), next r2 in B2 part (cols p1..p-1)
    # V1 has shape (m*p1, r1), V2 has shape (m*p2, r2)
    # V_combined[j*p+k, c]: for c<r1, k<p1: V1[j*p1+k, c]
    # for c>=r1, k>=p1: V2[j*p2+(k-p1), c-r1]
    for j in range(m):
        V[j*p:(j+1)*p1, :r1] = V1[j*p1:(j+1)*p1, :]
        V[j*p+p1:(j+1)*p, r1:] = V2[j*p2:(j+1)*p2, :]
    
    # W: W1 has shape (p1*n, r1), W2 has shape (p2*n, r2)
    # W_combined[k*n+i, c]: for c<r1, k<p1: W1[k*n+i, c]
    # for c>=r1, k>=p1: W2[(k-p1)*n+i, c-r1]
    for k in range(p):
        if k < p1:
            W[k*n:(k+1)*n, :r1] = W1[k*n:(k+1)*n, :]
            W[k*n:(k+1)*n, r1:] = 0
        else:
            W[k*n:(k+1)*n, :r1] = 0
            W[k*n:(k+1)*n, r1:] = W2[(k-p1)*n:(k-p1+1)*n, :]
    
    return U, V, W
