DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    r = 36
    U = np.zeros((n*m, r), dtype=int)
    V = np.zeros((m*p, r), dtype=int)
    W = np.zeros((p*n, r), dtype=int)
    for idx in range(r):
        i = (idx // (m*p)) % n
        j = (idx // p) % m
        k = idx % p
        U[i*m+j, idx] = 1
        V[j*p+k, idx] = 1
        W[k*n+i, idx] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
