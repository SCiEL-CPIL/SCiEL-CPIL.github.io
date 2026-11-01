DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 23
    # ... define U23, V23, W23 ...
    # Verify
    if verify(U23, V23, W23, n, m, p):
        return {"U": U23, "V": V23, "W": W23}
    else:
        # Return rank 27
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    # Check if sum U V W == Tensor of all 1s
    # Tensor T[i,j,k] = 1
    # Sum over r of U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k
    # Note: W index is k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True
