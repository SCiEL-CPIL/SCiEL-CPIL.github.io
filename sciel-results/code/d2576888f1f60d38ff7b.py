import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get schoolbook decomposition
    def schoolbook(n, m, p):
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

    # Helper to get Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        rank = 7
        U = np.zeros((2 * 2, rank))
        V = np.zeros((2 * 2, rank))
        W = np.zeros((2 * 2, rank))
        
        # M1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3: A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4: A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Recursive decomposition
    def get_decomp(n, m, p):
        # Base case: small dimensions, use schoolbook
        if n == 1 and m == 1 and p == 1:
            return schoolbook(1, 1, 1)
        if n * m * p <= 8:
            return schoolbook(n, m, p)
            
        # Specific optimizations
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
            
        # Additive splits
        # Try splitting m
        if m >= 2:
            # Split m into 2 and m-2
            if m == 5:
                n1, m1, p1 = n, 2, p
                n2, m2, p2 = n, 3, p
                U1, V1, W1 = get_decomp(n1, m1, p1)
                U2, V2, W2 = get_decomp(n2, m2, p2)
                U = np.hstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2])
                return U, V, W
            elif m == 3:
                # Split 3 into 2 + 1
                U1, V1, W1 = get_decomp(n, 2, p)
                U2, V2, W2 = get_decomp(n, 1, p)
                U = np.hstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2])
                return U, V, W
            elif m == 4:
                U1, V1, W1 = get_decomp(n, 2, p)
                U2, V2, W2 = get_decomp(n, 2, p)
                U = np.hstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2])
                return U, V, W

        # Try splitting p
        if p >= 2:
            if p == 7:
                # Split 7 into 2+2+2+1
                U_list = []
                V_list = []
                W_list = []
                for _ in range(3):
                    Ui, Vi, Wi = get_decomp(n, m, 2)
                    U_list.append(Ui)
                    V_list.append(Vi)
                    W_list.append(Wi)
                Ui, Vi, Wi = get_decomp(n, m, 1)
                U_list.append(Ui)
                V_list.append(Vi)
                W_list.append(Wi)
                U = np.hstack(U_list)
                V = np.hstack(V_list)
                W = np.hstack(W_list)
                return U, V, W
            elif p == 5:
                U1, V1, W1 = get_decomp(n, m, 2)
                U2, V2, W2 = get_decomp(n, m, 3)
                U = np.hstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2])
                return U, V, W
            elif p == 3:
                U1, V1, W1 = get_decomp(n, m, 2)
                U2, V2, W2 = get_decomp(n, m, 1)
                U = np.hstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2])
                return U, V, W

        # Try splitting n
        if n >= 2:
            if n == 3:
                U1, V1, W1 = get_decomp(2, m, p)
                U2, V2, W2 = get_decomp(1, m, p)
                U = np.hstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2])
                return U, V, W

        # Fallback to schoolbook
        return schoolbook(n, m, p)

    U, V, W = get_decomp(n, m, p)
    
    # Convert to lists of lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
