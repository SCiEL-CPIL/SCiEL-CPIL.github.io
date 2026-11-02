import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_decomp(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return np.array([[1]]), np.array([[1]]), np.array([[1]])
        if p == 1:
            r = n * m
            U = np.zeros((n * m, r))
            V = np.zeros((m, r))
            W = np.zeros((n, r))
            for i in range(n):
                for j in range(m):
                    c = i * m + j
                    U[i * m + j, c] = 1
                    V[j, c] = 1
                    W[i, c] = 1
            return U, V, W
        if n == 1:
            r = m * p
            U = np.zeros((m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p, r))
            for j in range(m):
                for k in range(p):
                    c = j * p + k
                    U[j, c] = 1
                    V[j * p + k, c] = 1
                    W[k, c] = 1
            return U, V, W
        if m == 1:
            r = n * p
            U = np.zeros((n, r))
            V = np.zeros((p, r))
            W = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    c = i * p + k
                    U[i, c] = 1
                    V[k, c] = 1
                    W[k * n + i, c] = 1
            return U, V, W
            
        # Strassen case for even dimensions
        if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
            n1, n2 = n // 2, n // 2
            m1, m2 = m // 2, m // 2
            p1, p2 = p // 2, p // 2
            
            # Get 7 sub-decompositions
            # M1: (A11+A22)(B11+B22)
            # This is complex to combine recursively without a generic linear algebra framework.
            # For the specific case 2x4x2, we can hardcode the 14 rank terms.
            
            # For now, fall back to splitting p
            pass

        # Split p
        p1 = p // 2
        p2 = p - p1
        
        U1, V1, W1 = get_decomp(n, m, p1)
        U2, V2, W2 = get_decomp(n, m, p2)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        U_total = np.zeros((n * m, r1 + r2))
        V_total = np.zeros((m * p, r1 + r2))
        W_total = np.zeros((p * n, r1 + r2))
        
        U_total[:, :r1] = U1
        V_total[:, :r1] = V1
        W_total[:p1, :r1] = W1
        
        U_total[:, r1:] = U2
        V_total[:, r1:] = V2
        W_total[p1:, r1:] = W2
        
        return U_total, V_total, W_total

    U_np, V_np, W_np = get_decomp(n, m, p)
    
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
# EVOLVE-BLOCK-END
