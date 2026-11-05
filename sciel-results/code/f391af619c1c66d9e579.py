import numpy as np

def get_factors(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int)
    
    if n == 2 and m == 2 and p == 2:
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    if n == 1:
        rank = m * p
        U = np.zeros((m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p, rank), dtype=int)
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j * p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W

    if p % 2 == 0:
        p1, p2 = p // 2, p // 2
        U1, V1, W1 = get_factors(n, m, p1)
        U2, V2, W2 = get_factors(n, m, p2)
        U = np.hstack([U1, U2])
        V = np.block([[V1, np.zeros((V1.shape[0], V2.shape[1]), dtype=int)], 
                      [np.zeros((V2.shape[0], V1.shape[1]), dtype=int), V2]])
        W = np.block([[W1, np.zeros((W1.shape[0], W2.shape[1]), dtype=int)], 
                      [np.zeros((W2.shape[0], W1.shape[1]), dtype=int), W2]])
        return U, V, W
    elif m % 2 == 0:
        m1, m2 = m // 2, m // 2
        U1, V1, W1 = get_factors(n, m1, p)
        U2, V2, W2 = get_factors(n, m2, p)
        U = np.block([[U1, np.zeros((U1.shape[0], U2.shape[1]), dtype=int)], 
                      [np.zeros((U2.shape[0], U1.shape[1]), dtype=int), U2]])
        V = np.block([[V1, np.zeros((V1.shape[0], V2.shape[1]), dtype=int)], 
                      [np.zeros((V2.shape[0], V1.shape[1]), dtype=int), V2]])
        W = np.hstack([W1, W2])
        return U, V, W
    elif n % 2 == 0:
        n1, n2 = n // 2, n // 2
        U1, V1, W1 = get_factors(n1, m, p)
        U2, V2, W2 = get_factors(n2, m, p)
        U = np.block([[U1, np.zeros((U1.shape[0], U2.shape[1]), dtype=int)], 
                      [np.zeros((U2.shape[0], U1.shape[1]), dtype=int), U2]])
        V = np.hstack([V1, V2])
        W = np.block([[W1, np.zeros((W1.shape[0], W2.shape[1]), dtype=int)], 
                      [np.zeros((W2.shape[0], W1.shape[1]), dtype=int), W2]])
        return U, V, W
    else:
        n1, n2 = 1, n - 1
        U1, V1, W1 = get_factors(n1, m, p)
        U2, V2, W2 = get_factors(n2, m, p)
        U = np.block([[U1, np.zeros((U1.shape[0], U2.shape[1]), dtype=int)], 
                      [np.zeros((U2.shape[0], U1.shape[1]), dtype=int), U2]])
        V = np.hstack([V1, V2])
        W = np.block([[W1, np.zeros((W1.shape[0], W2.shape[1]), dtype=int)], 
                      [np.zeros((W2.shape[0], W1.shape[1]), dtype=int), W2]])
        return U, V, W

def construct():
    n, m, p = 3, 4, 8
    U, V, W = get_factors(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
