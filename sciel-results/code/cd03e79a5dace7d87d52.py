import numpy as np

def check_independence(U, V, W, n, m, p):
    rank = len(U[0])
    # Flatten each rank-1 tensor u_r \otimes v_r \otimes w_r
    # Size: (n*m) * (m*p) * (p*n)
    dim = (n*m) * (m*p) * (p*n)
    tensors = []
    for r in range(rank):
        t = np.zeros(dim)
        for a in range(n*m):
            if U[a][r] != 0:
                for b in range(m*p):
                    if V[b][r] != 0:
                        for c in range(p*n):
                            if W[c][r] != 0:
                                idx = a * (m*p) * (p*n) + b * (p*n) + c
                                t[idx] = U[a][r] * V[b][r] * W[c][r]
        tensors.append(t)
    
    T = np.array(tensors).T # Shape (dim, rank)
    # Compute rank
    r = np.linalg.matrix_rank(T)
    return r

# In construct:
if check_independence(U, V, W, n, m, p) < 56:
    # Try to reduce?
    pass
