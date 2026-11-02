import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Recursive decomposition function
    def decompose(n, m, p):
        # Returns U, V, W for n x m x p
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition
            # U, V, W shapes: (4, 7), (4, 7), (4, 7)
            # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
            # i in 0..1, j in 0..1, k in 0..1
            # Strassen factors:
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B12 - B22)
            # M4 = A22(B21 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # U indices: 0:00, 1:01, 2:10, 3:11
            # V indices: 0:00, 1:01, 2:10, 3:11
            # W indices: 0:00, 1:01, 2:10, 3:11
            
            # M1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1
            # U: A00, A11
            U[0][0] = 1; U[3][0] = 1
            # V: B00, B11
            V[0][0] = 1; V[3][0] = 1
            # W: C00, C01, C10, C11
            W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
            
            # M2: (A10+A11)B00 -> C10 += M2, C11 += M2
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = 1
            
            # M3: A00(B01-B11) -> C00 -= M3, C01 += M3
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[0][2] = -1; W[1][2] = 1
            
            # M4: A11(B10-B00) -> C10 += M4, C11 -= M4
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[2][3] = 1; W[3][3] = -1
            
            # M5: (A00+A01)B11 -> C01 += M5, C11 += M5
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[1][4] = 1; W[3][4] = 1
            
            # M6: (A10-A00)(B00+B01) -> C10 += M6, C00 -= M6
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[2][5] = 1; W[0][5] = -1
            
            # M7: (A01-A11)(B10+B11) -> C01 += M7, C11 -= M7
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[1][6] = 1; W[3][6] = -1
            
            return U, V, W

        if n == 1:
            # 1 x m x p
            # C[0, k] = sum_j A[0, j] B[j, k]
            # This is dot product. Rank m?
            # Actually, we can use schoolbook rank m*p?
            # Or better:
            # If m=1, rank p?
            # Let's use schoolbook for simplicity in this branch if not optimized.
            rank = m * p
            U = [[0]*rank for _ in range(m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p)]
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j][col] = 1
                    V[j*p+k][col] = 1
                    W[k][col] = 1
                    col += 1
            return U, V, W

        if m == 1:
            # n x 1 x p
            # C[i, k] = A[i, 0] B[0, k]
            # Outer product. Rank 2?
            # C[i,k] = A[i,0] B[0,k]
            # U[i, 0] = A[i,0] -> U[i,0]=1
            # V[0, 0] = B[0,k] -> V[0,0]=1
            # W[k, i, 0] = 1
            U = [[0]*2 for _ in range(n)]
            V = [[0]*2 for _ in range(p)]
            W = [[0]*2 for _ in range(p*n)]
            for i in range(n):
                U[i][0] = 1
            V[0][1] = 1
            for k in range(p):
                for i in range(n):
                    W[k*n+i][1] = 1
            # Wait, U[i,0] V[0,1] W[k*n+i, 1] = 1 * 1 * 1 = 1.
            # This gives C[i,k] = 1. We need A[i,0] B[0,k].
            # The decomposition is linear.
            # C[i,k] = sum_r U[i,0,r] V[0,k,r] W[k,i,r]
            # We need U[i,0,r] V[0,k,r] W[k,i,r] to sum to A[i,0] B[0,k].
            # This is not possible with constant U,V,W unless we fix them.
            # The tensor is T[i, 0, k] = 1.
            # Decomposition: sum_r U[i,0,r] V[0,k,r] W[k,i,r] = 1.
            # Rank 2:
            # r=0: U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1 -> Sum = 1.
            # This works for the tensor T[i,0,k]=1.
            # The actual multiplication is C[i,k] = A[i,0] B[0,k].
            # The tensor T is such that C[i,k] = <T, A_i0 B_0k>.
            # T[i,0,k] = 1.
            # So rank 2 is correct for the tensor.
            U = [[0]*2 for _ in range(n)]
            V = [[0]*2 for _ in range(p)]
            W = [[0]*2 for _ in range(p*n)]
            for i in range(n):
                U[i][0] = 1
            for k in range(p):
                V[0*k][1] = 1 # V index is j*p+k = 0*p+k = k
            for k in range(p):
                for i in range(n):
                    W[k*n+i][0] = 1
            # Check: U[i,0]=1, V[k,1]=1, W[ki,0]=1.
            # Product: 1*1*1 = 1.
            # This is correct for T[i,0,k]=1.
            return U, V, W

        if p == 1:
            # n x m x 1
            # C[i, 0] = sum_j A[i,j] B[j,0]
            # T[i,j,0] = 1.
            # Rank m?
            # U[i,j,r], V[j,0,r], W[0,i,r]
            # Schoolbook rank m.
            rank = m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m)]
            W = [[0]*rank for _ in range(n)]
            for j in range(m):
                for i in range(n):
                    U[i*m+j][j] = 1
                V[j][j] = 1
                W[i][j] = 1 # Wait, W index is k*n+i = 0*n+i = i.
            # Correct W indexing:
            W = [[0]*rank for _ in range(n)]
            for j in range(m):
                V[j][j] = 1
                for i in range(n):
                    U[i*m+j][j] = 1
                    W[i][j] = 1
            return U, V, W

        # Recursive case
        # Split largest dimension
        if m >= n and m >= p:
            m1 = m // 2
            m2 = m - m1
            # A: n x m = [A1 (n x m1), A2 (n x m2)]
            # B: m x p = [B1 (m1 x p); B2 (m2 x p)]
            # C: n x p = A1 B1 + A2 B2
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            
            rank = len(U1[0]) + len(U2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Copy U1
            for i in range(n):
                for j in range(m1):
                    for r in range(len(U1[0])):
                        U[i*m+j][r] = U1[i*m1+j][r]
            # Copy U2 (offset in m)
            for i in range(n):
                for j in range(m2):
                    for r in range(len(U2[0])):
                        U[i*m + (m1+j)][len(U1[0])+r] = U2[i*m2+j][r]
                        
            # Copy V1
            for j in range(m1):
                for k in range(p):
                    for r in range(len(V1[0])):
                        V[j*p+k][r] = V1[j*p+k][r]
            # Copy V2 (offset in m)
            for j in range(m2):
                for k in range(p):
                    for r in range(len(V2[0])):
                        V[(m1+j)*p+k][len(V1[0])+r] = V2[j*p+k][r]
                        
            # Copy W
            for k in range(p):
                for i in range(n):
                    for r in range(len(W1[0])):
                        W[k*n+i][r] = W1[k*n+i][r]
                        W[k*n+i][len(W1[0])+r] = W2[k*n+i][r]
                        
            return U, V, W
            
        elif p >= n and p >= m:
            p1 = p // 2
            p2 = p - p1
            # A: n x m
            # B: m x p = [B1 (m x p1), B2 (m x p2)]
            # C: n x p = [C1 (n x p1), C2 (n x p2)]
            # C1 = A B1, C2 = A B2
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            
            rank = len(U1[0]) + len(U2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Copy U
            for i in range(n):
                for j in range(m):
                    for r in range(len(U1[0])):
                        U[i*m+j][r] = U1[i*m+j][r]
                        U[i*m+j][len(U1[0])+r] = U2[i*m+j][r]
                        
            # Copy V1
            for j in range(m):
                for k in range(p1):
                    for r in range(len(V1[0])):
                        V[j*p+k][r] = V1[j*p1+k][r]
            # Copy V2
            for j in range(m):
                for k in range(p2):
                    for r in range(len(V2[0])):
                        V[j*p + (p1+k)][len(V1[0])+r] = V2[j*p2+k][r]
                        
            # Copy W1
            for k in range(p1):
                for i in range(n):
                    for r in range(len(W1[0])):
                        W[k*n+i][r] = W1[k*p1+i][r] # W index in subproblem is k_sub * n + i
            # Copy W2
            for k in range(p2):
                for i in range(n):
                    for r in range(len(W2[0])):
                        W[(p1+k)*n+i][len(W1[0])+r] = W2[k*p2+i][r]
                        
            return U, V, W
            
        else:
            # Split n
            n1 = n // 2
            n2 = n - n1
            # A: n x m = [A1 (n1 x m); A2 (n2 x m)]
            # B: m x p
            # C: n x p = [C1 (n1 x p); C2 (n2 x p)]
            # C1 = A1 B, C2 = A2 B
            U1, V1, W1 = decompose(n1, m, p)
            U2, V2, W2 = decompose(n2, m, p)
            
            rank = len(U1[0]) + len(U2[0])
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Copy U1
            for i in range(n1):
                for j in range(m):
                    for r in range(len(U1[0])):
                        U[i*m+j][r] = U1[i*m+j][r]
            # Copy U2
            for i in range(n2):
                for j in range(m):
                    for r in range(len(U2[0])):
                        U[(n1+i)*m+j][len(U1[0])+r] = U2[i*m+j][r]
                        
            # Copy V
            for j in range(m):
                for k in range(p):
                    for r in range(len(V1[0])):
                        V[j*p+k][r] = V1[j*p+k][r]
                        V[j*p+k][len(V1[0])+r] = V2[j*p+k][r]
                        
            # Copy W1
            for k in range(p):
                for i in range(n1):
                    for r in range(len(W1[0])):
                        W[k*n+i][r] = W1[k*n1+i][r]
            # Copy W2
            for k in range(p):
                for i in range(n2):
                    for r in range(len(W2[0])):
                        W[k*n + (n1+i)][len(W1[0])+r] = W2[k*n2+i][r]
                        
            return U, V, W

    U, V, W = decompose(n, m, p)
    
    # Verify
    rank = len(U[0])
    # Construct tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = np.array(U[:, r])
        v = np.array(V[:, r])
        w = np.array(W[:, r])
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    # Expected tensor
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
                
    if not np.allclose(T, T_exp):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
