import numpy as np

def construct():
    n, m, p = 3, 5, 7
    
    def get_decomp(n, m, p):
        # Returns U, V, W as lists of lists
        # U: n*m rows, rank cols
        # V: m*p rows, rank cols
        # W: p*n rows, rank cols
        
        if n == 1:
            rank = m * p
            U = [[0]*rank for _ in range(m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p)]
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    U[j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k][idx] = 1
            return U, V, W
            
        if m == 1:
            rank = n * p
            U = [[0]*rank for _ in range(n)]
            V = [[0]*rank for _ in range(p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for k in range(p):
                    idx = i * p + k
                    U[i][idx] = 1
                    V[k][idx] = 1
                    W[k * n + i][idx] = 1
            return U, V, W
            
        if p == 1:
            rank = n * m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m)]
            W = [[0]*rank for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    U[i * m + j][idx] = 1
                    V[j][idx] = 1
                    W[i][idx] = 1
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            # Strassen
            U = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
            # Need 7 columns
            # M1: A11(B11-B21)
            # M2: (A11+A12)B21
            # M3: (A21+A22)B12
            # M4: A22(B22-B12)
            # M5: (A11+A22)(B11+B22)
            # M6: (A12-A22)(B21+B22)
            # M7: (A11+A21)(B11-B12)
            
            U = [
                [1, 0, 0, 0], # M1
                [1, 1, 0, 0], # M2
                [0, 0, 1, 1], # M3
                [0, 0, 0, 1], # M4
                [1, 0, 0, 1], # M5
                [0, 1, 0, -1],# M6
                [1, 0, 1, 0]  # M7
            ]
            # Wait, U rows are A11, A12, A21, A22.
            # M1: A11. U=[1,0,0,0]
            # M2: A11+A12. U=[1,1,0,0]
            # M3: A21+A22. U=[0,0,1,1]
            # M4: A22. U=[0,0,0,1]
            # M5: A11+A22. U=[1,0,0,1]
            # M6: A12-A22. U=[0,1,0,-1]
            # M7: A11+A21. U=[1,0,1,0]
            
            V = [
                [1, 0, -1, 0], # B11-B21
                [0, 0, 1, 0],  # B21
                [0, 1, 0, 0],  # B12
                [0, 0, 0, 1],  # B22-B12 -> [0, -1, 0, 1]? No, B22-B12 is [0,-1,0,1] if B is B11 B12 B21 B22.
                # B index: 0=B11, 1=B12, 2=B21, 3=B22.
                # M1: B11-B21 -> [1, 0, -1, 0]
                # M2: B21 -> [0, 0, 1, 0]
                # M3: B12 -> [0, 1, 0, 0]
                # M4: B22-B12 -> [0, -1, 0, 1]
                # M5: B11+B22 -> [1, 0, 0, 1]
                # M6: B21+B22 -> [0, 0, 1, 1]
                # M7: B11-B12 -> [1, -1, 0, 0]
            ]
            V = [
                [1, 0, -1, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, -1, 0, 0]
            ]
            
            W = [
                [1, 0, 0, 0], # C11
                [0, 1, 0, 0], # C12
                [0, 0, 1, 0], # C21
                [0, 0, 0, 1]  # C22
            ]
            # C index: 0=C11, 1=C12, 2=C21, 3=C22.
            # M1: C11+1, C22-1?
            # C11 = M1+M4-M5+M7
            # C12 = M2+M3
            # C21 = M6+M7
            # C22 = M1-M3+M5+M6
            
            W = [
                [1, 0, 0, -1], # M1: C11=1, C22=-1
                [0, 1, 0, 0],  # M2: C12=1
                [0, 1, 0, -1], # M3: C12=1, C22=-1
                [1, 0, 0, 0],  # M4: C11=1
                [-1, 0, 0, 1], # M5: C11=-1, C22=1
                [0, 0, 1, 1],  # M6: C21=1, C22=1
                [1, 0, 1, 0]   # M7: C11=1, C21=1
            ]
            
            # Verify M1: A11(B11-B21).
            # U=[1,0,0,0], V=[1,0,-1,0].
            # Product: A11 * (B11 - B21).
            # W=[1,0,0,-1].
            # C11 += A11(B11-B21). C22 -= A11(B11-B21).
            # Formula: C11 = M1 + ... C22 = M1 - ...
            # Yes.
            
            return U, V, W
            
        # Recursive step
        # Split largest dimension
        if n >= m and n >= p:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # Fill U1
            for i in range(n1*m):
                for c in range(rank1):
                    U[i][c] = U1[i][c]
            # Fill U2
            for i in range(n2*m):
                for c in range(rank2):
                    U[n1*m + i][rank1 + c] = U2[i][c]
            # Fill V
            for i in range(m*p):
                for c in range(rank1):
                    V[i][c] = V1[i][c]
                for c in range(rank2):
                    V[i][rank1 + c] = V2[i][c]
            # Fill W
            for i in range(p*n):
                for c in range(rank1):
                    W[i][c] = W1[i][c]
                for c in range(rank2):
                    W[i][rank1 + c] = W2[i][c]
                    
            return U, V, W
        elif m >= n and m >= p:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # U: rows i*m1+j
            for i in range(n):
                for j in range(m1):
                    for c in range(rank1):
                        U[i*m + j][c] = U1[i*m1 + j][c]
                for j in range(m2):
                    for c in range(rank2):
                        U[i*m + m1 + j][rank1 + c] = U2[i*m2 + j][c]
                        
            # V: rows j*p+k
            for j in range(m1):
                for k in range(p):
                    for c in range(rank1):
                        V[j*p + k][c] = V1[j*p + k][c]
            for j in range(m2):
                for k in range(p):
                    for c in range(rank2):
                        V[(m1+j)*p + k][rank1 + c] = V2[j*p + k][c]
                        
            # W: same as before
            for i in range(p*n):
                for c in range(rank1):
                    W[i][c] = W1[i][c]
                for c in range(rank2):
                    W[i][rank1 + c] = W2[i][c]
                    
            return U, V, W
        else:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # U: same
            for i in range(n*m):
                for c in range(rank1):
                    U[i][c] = U1[i][c]
                for c in range(rank2):
                    U[i][rank1 + c] = U2[i][c]
                    
            # V: rows j*p1+k
            for j in range(m):
                for k in range(p1):
                    for c in range(rank1):
                        V[j*p + k][c] = V1[j*p1 + k][c]
                for k in range(p2):
                    for c in range(rank2):
                        V[j*p + p1 + k][rank1 + c] = V2[j*p2 + k][c]
                        
            # W: rows k*n+i
            for k in range(p1):
                for i in range(n):
                    for c in range(rank1):
                        W[k*n + i][c] = W1[k*n + i][c]
            for k in range(p2):
                for i in range(n):
                    for c in range(rank2):
                        W[(p1+k)*n + i][rank1 + c] = W2[k*n + i][c]
                        
            return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
