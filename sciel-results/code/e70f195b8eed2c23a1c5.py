import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zero_mat(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Helper to add two matrices (element-wise)
    def add_mat(A, B):
        rows = len(A)
        cols = len(A[0])
        return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]
    
    # Helper to scale a matrix
    def scale_mat(A, s):
        return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]
    
    # Schoolbook decomposition for M_{n,m,p}
    # Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p)
    def schoolbook(n, m, p):
        rank = n * m * p
        U = zero_mat(n * m, rank)
        V = zero_mat(m * p, rank)
        W = zero_mat(p * n, rank)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        return U, V, W
    
    # Strassen decomposition for M_{2,2,2}
    # Returns U (4x7), V (4x7), W (4x7)
    def strassen_2x2x2():
        n, m, p = 2, 2, 2
        rank = 7
        U = zero_mat(n * m, rank)
        V = zero_mat(m * p, rank)
        W = zero_mat(p * n, rank)
        
        # Indices:
        # U: i*m+j -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V: j*p+k -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W: k*n+i -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # m1 = (a+d)(e+h)
        # a=U[0], d=U[3], e=V[0], h=V[3]
        # C00 += 1, C11 += 1 -> W[0]=1, W[3]=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # m2 = (c+d)e
        # c=U[2], d=U[3], e=V[0]
        # C10 += 1, C11 += -1 -> W[2]=1, W[3]=-1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # m3 = a(f-h)
        # a=U[0], f=V[1], h=V[3]
        # C01 += 1, C11 += 1 -> W[1]=1, W[3]=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # m4 = d(g-e)
        # d=U[3], g=V[2], e=V[0]
        # C00 += 1, C10 += 1 -> W[0]=1, W[2]=1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # m5 = (a+b)h
        # a=U[0], b=U[1], h=V[3]
        # C00 += -1, C01 += 1 -> W[0]=-1, W[1]=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # m6 = (c-a)(e+f)
        # c=U[2], a=U[0], e=V[0], f=V[1]
        # C11 += 1 -> W[3]=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # m7 = (b-d)(g+h)
        # b=U[1], d=U[3], g=V[2], h=V[3]
        # C00 += 1, C10 += -1 -> W[0]=1, W[2]=-1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[2][6] = -1
        
        return U, V, W

    # Recursive construction with index mapping
    # n, m, p: local dimensions
    # i_off, j_off, k_off: offsets in global i, j, k
    # Returns U (n*m x r), V (m*p x r), W (p*n x r)
    def build(n, m, p):
        # Base case: Use Strassen if 2x2x2
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # Otherwise, use Schoolbook for simplicity and correctness in base cases
        # But we can try to split to reduce rank.
        # Strategy: Split p if p > 1, else split m if m > 1, else schoolbook.
        # Actually, for M_{2,2,p}, splitting p into 2s is good.
        # For M_{2,3,p}, splitting m into 2+1 is good.
        
        # Heuristic:
        # If n==2 and m==2:
        #    If p == 2: Strassen
        #    Else: Split p into p1, p2 such that p1 is as close to 2 as possible?
        #           Or just split p into 2 and p-2?
        # If n==2 and m==3:
        #    Split m into 2 and 1.
        
        # Let's implement the specific splits derived earlier.
        
        if n == 2 and m == 2:
            if p == 2:
                return strassen_2x2x2()
            else:
                # Split p into 2 and p-2
                p1 = 2
                p2 = p - 2
                U1, V1, W1 = build(n, m, p1)
                U2, V2, W2 = build(n, m, p2)
                
                # Map indices
                # U is same
                # V: j*p+k. For block 1, k in [0, p1-1]. For block 2, k in [p1, p-1].
                # W: k*n+i. For block 1, k in [0, p1-1]. For block 2, k in [p1, p-1].
                
                # Concatenate
                r1 = len(U1[0])
                r2 = len(U2[0])
                r = r1 + r2
                
                U = [U1[i] + U2[i] for i in range(n*m)]
                
                V = []
                for i in range(m*p):
                    j = i // p
                    k = i % p
                    if k < p1:
                        v_idx = j * p1 + k
                        V.append(V1[j * p1 + k] + [0]*r2)
                    else:
                        k2 = k - p1
                        v_idx = j * p2 + k2
                        V.append([0]*r1 + V2[j * p2 + k2])
                        
                W = []
                for i in range(p*n):
                    k = i // n
                    ii = i % n
                    if k < p1:
                        w_idx = k * n + ii
                        W.append(W1[k * n + ii] + [0]*r2)
                    else:
                        k2 = k - p1
                        w_idx = k2 * n + ii
                        W.append([0]*r1 + W2[k2 * n + ii])
                        
                return U, V, W

        elif n == 2 and m == 3:
            # Split m into 2 and 1
            m1 = 2
            m2 = 1
            U1, V1, W1 = build(n, m1, p)
            U2, V2, W2 = build(n, m2, p)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            r = r1 + r2
            
            U = []
            for i in range(n*m):
                ii = i // m
                jj = i % m
                if jj < m1:
                    u_idx = ii * m1 + jj
                    U.append(U1[ii * m1 + jj] + [0]*r2)
                else:
                    jj2 = jj - m1
                    u_idx = ii * m2 + jj2
                    U.append([0]*r1 + U2[ii * m2 + jj2])
                    
            V = []
            for i in range(m*p):
                jj = i // p
                kk = i % p
                if jj < m1:
                    v_idx = jj * p + kk
                    V.append(V1[jj * p + kk] + [0]*r2)
                else:
                    jj2 = jj - m1
                    v_idx = jj2 * p + kk
                    V.append([0]*r1 + V2[jj2 * p + kk])
                    
            # W is same
            W = [W1[i] + W2[i] for i in range(p*n)]
            
            return U, V, W
            
        else:
            # Fallback to schoolbook
            return schoolbook(n, m, p)

    U, V, W = build(n, m, p)
    
    # Convert to list of lists of integers if possible
    # The values are integers in this construction.
    
    return {"U": U, "V": V, "W": W}
