import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    def solve(n, m, p):
        # Base cases
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return U, V, W

        if n == 2 and m == 2 and p == 2:
            # Strassen
            rank = 7
            U = np.zeros((4, rank))
            V = np.zeros((4, rank))
            W = np.zeros((4, rank))
            
            # M1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = 1
            
            # M3
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            
            # M4
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            
            # M5
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            
            # M6
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # M7
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1; W[2, 6] = -1
            
            return U, V, W

        # Recursive step
        # Split m if possible
        if m >= 2:
            if m > 2:
                m1 = 2
                m2 = m - 2
            else:
                # m == 2
                if p > 2:
                    # Split p
                    p1 = 2
                    p2 = p - 2
                    U1, V1, W1 = solve(n, m, p1)
                    U2, V2, W2 = solve(n, m, p2)
                    
                    R1 = U1.shape[1]
                    R2 = U2.shape[1]
                    R = R1 + R2
                    
                    U = np.concatenate([U1, U2], axis=1)
                    
                    V = np.zeros((m * p, R))
                    # V1 part
                    for j in range(m):
                        for k in range(p1):
                            V[j * p + k, :R1] = V1[j * p1 + k, :]
                    # V2 part
                    for j in range(m):
                        for k in range(p2):
                            V[j * p + (p1 + k), R1:] = V2[j * p2 + k, :]
                            
                    W = np.zeros((p * n, R))
                    # W1 part
                    for k in range(p1):
                        for i in range(n):
                            W[k * n + i, :R1] = W1[k * n + i, :]
                    # W2 part
                    for k in range(p2):
                        for i in range(n):
                            W[(p1 + k) * n + i, R1:] = W2[k * n + i, :]
                            
                    return U, V, W
                else:
                    # p <= 2, handled by base cases or Strassen
                    # If p==2 and m==2 and n==2, Strassen handled above.
                    # If p==1, base case handled above.
                    # So this branch shouldn't be reached for m==2, p<=2
                    pass
            
            # Split m
            U1, V1, W1 = solve(n, m1, p)
            U2, V2, W2 = solve(n, m2, p)
            
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R = R1 + R2
            
            U = np.zeros((n * m, R))
            # U1 part
            for i in range(n):
                for j in range(m1):
                    U[i * m + j, :R1] = U1[i * m1 + j, :]
            # U2 part
            for i in range(n):
                for j in range(m2):
                    U[i * m + (m1 + j), R1:] = U2[i * m2 + j, :]
                    
            V = np.zeros((m * p, R))
            # V1 part
            for j in range(m1):
                for k in range(p):
                    V[j * p + k, :R1] = V1[j * p + k, :]
            # V2 part
            for j in range(m2):
                for k in range(p):
                    V[(m1 + j) * p + k, R1:] = V2[j * p + k, :]
                    
            W = np.concatenate([W1, W2], axis=1)
            
            return U, V, W
        else:
            # m == 1
            if p > 2:
                p1 = 2
                p2 = p - 2
                U1, V1, W1 = solve(n, m, p1)
                U2, V2, W2 = solve(n, m, p2)
                
                R1 = U1.shape[1]
                R2 = U2.shape[1]
                R = R1 + R2
                
                U = np.concatenate([U1, U2], axis=1)
                
                V = np.zeros((m * p, R))
                # V1 part
                for j in range(m):
                    for k in range(p1):
                        V[j * p + k, :R1] = V1[j * p1 + k, :]
                # V2 part
                for j in range(m):
                    for k in range(p2):
                        V[j * p + (p1 + k), R1:] = V2[j * p2 + k, :]
                        
                W = np.zeros((p * n, R))
                # W1 part
                for k in range(p1):
                    for i in range(n):
                        W[k * n + i, :R1] = W1[k * n + i, :]
                # W2 part
                for k in range(p2):
                    for i in range(n):
                        W[(p1 + k) * n + i, R1:] = W2[k * n + i, :]
                        
                return U, V, W
            else:
                # Should be handled by base cases
                return solve(n, m, p) # Fallback, but shouldn't happen

    U, V, W = solve(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
