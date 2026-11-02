import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    def get_decomp(n, m, p):
        # Returns U (n*m, R), V (m*p, R), W (p*n, R)
        if m == 1:
            # Rank 1
            R = 1
            U = np.zeros((n, R))
            V = np.zeros((p, R))
            W = np.zeros((p*n, R))
            U[:, 0] = 1
            V[:, 0] = 1
            W[:, 0] = 1
            return U, V, W
        if n == 1:
            # Rank m*p
            R = m * p
            U = np.zeros((m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p, R))
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    U[j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k, idx] = 1
            return U, V, W
        if p == 1:
            # Rank n*m
            R = n * m
            U = np.zeros((n*m, R))
            V = np.zeros((m, R))
            W = np.zeros((n, R))
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    U[i * m + j, idx] = 1
                    V[j, idx] = 1
                    W[i, idx] = 1
            return U, V, W
            
        # Try splitting m
        # Heuristic: split m into 1 and m-1, or 2 and m-2, etc.
        # We'll try a few splits and pick the one with min rank.
        best_rank = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R = R1 + R2
            if R < best_rank:
                # Embed
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                
                # Sub 1
                U[:n*m1, :R1] = U1
                V[:m1*p, :R1] = V1
                W[:, :R1] = W1
                
                # Sub 2
                U[n*m1:, R1:] = U2
                V[m1*p:, R1:] = V2
                W[:, R1:] = W2
                
                best_rank = R
                best_U, best_V, best_W = U, V, W

        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R = R1 + R2
            if R < best_rank:
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                
                # Sub 1
                U[:n1*m, :R1] = U1
                V[:, :R1] = V1
                W[:p*n1, :R1] = W1
                
                # Sub 2
                U[n1*m:, R1:] = U2
                V[:, R1:] = V2
                W[p*n1:, R1:] = W2
                
                best_rank = R
                best_U, best_V, best_W = U, V, W

        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R = R1 + R2
            if R < best_rank:
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                
                # Sub 1
                U[:, :R1] = U1
                V[:m*p1, :R1] = V1
                W[:p1*n, :R1] = W1
                
                # Sub 2
                U[:, R1:] = U2
                V[m*p1:, R1:] = V2
                W[p1*n:, R1:] = W2
                
                best_rank = R
                best_U, best_V, best_W = U, V, W

        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
