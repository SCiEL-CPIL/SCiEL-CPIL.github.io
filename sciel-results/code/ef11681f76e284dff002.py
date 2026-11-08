import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition recursively
    # Target rank is expected to be 50
    
    def get_decomp(n, m, p):
        # Returns U, V, W for M_{n,m,p}
        if m == 1:
            # Rank n*p
            rank = n * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i * m][col] = 1
                    V[k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            # Strassen
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            # M1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1
            
            # M3
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1
            
            # M4
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            
            # M5
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1
            
            # M6
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[3][6] = 1
            
            return U, V, W
        
        # Recursive case
        # Split m into m1, m2
        if m % 2 == 0:
            m1 = m // 2
            m2 = m // 2
        else:
            m1 = m // 2
            m2 = m - m1
            
        U1, V1, W1 = get_decomp(n, m1, p)
        U2, V2, W2 = get_decomp(n, m2, p)
        
        R1 = len(U1[0])
        R2 = len(U2[0])
        R = R1 + R2
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Embed U1
        for i in range(n):
            for j in range(m1):
                for c in range(R1):
                    U[i * m + j][c] = U1[i * m1 + j][c]
                    
        # Embed U2
        for i in range(n):
            for j in range(m2):
                for c in range(R2):
                    U[i * m + (m1 + j)][R1 + c] = U2[i * m2 + j][c]
                    
        # Embed V1
        for j in range(m1):
            for k in range(p):
                for c in range(R1):
                    V[j * p + k][c] = V1[j * p + k][c]
                    
        # Embed V2
        for j in range(m2):
            for k in range(p):
                for c in range(R2):
                    V[(m1 + j) * p + k][R1 + c] = V2[j * p + k][c]
                    
        # Embed W
        for k in range(p):
            for i in range(n):
                for c in range(R1):
                    W[k * n + i][c] = W1[k * n + i][c]
                for c in range(R2):
                    W[k * n + i][R1 + c] = W2[k * n + i][c]
                    
        return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
