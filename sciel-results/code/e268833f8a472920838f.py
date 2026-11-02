import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition recursively.
    # Returns U, V, W, rank
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    def get_decomp(n, m, p):
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1]])
            V = np.array([[1]])
            W = np.array([[1]])
            return U, V, W, 1
            
        # Base case: m=1 (Outer product structure)
        if m == 1:
            rank = n * p
            U = np.zeros((n * 1, rank))
            V = np.zeros((1 * p, rank))
            W = np.zeros((p * n, rank))
            idx = 0
            for i in range(n):
                for k in range(p):
                    U[i, idx] = 1
                    V[k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
            return U, V, W, rank

        # Base case: n=1, p=1 (Dot product)
        if n == 1 and p == 1:
            rank = m
            U = np.zeros((1 * m, rank))
            V = np.zeros((m * 1, rank))
            W = np.zeros((1 * 1, rank))
            for j in range(m):
                U[j, j] = 1
                V[j, j] = 1
                W[0, j] = 1
            return U, V, W, rank

        # Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            # U indices: i*2+j
            # V indices: j*2+k
            # W indices: k*2+i
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # Strassen M1..M7
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # We need to map these to U, V, W
            # C11 = M1 + M4 - M5 - M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M7
            
            # U corresponds to A. V to B. W to C.
            # The decomposition is sum of 7 rank 1 tensors.
            # Each tensor r has U_r, V_r, W_r.
            
            # r=0 (M1): A=[1,1;1,1] (scaled), B=[1,0;1,1]? No.
            # M1 = (A11+A22)(B11+B22)
            # U_0: A11=1, A12=0, A21=0, A22=1 -> [1,0,0,1]
            # V_0: B11=1, B12=0, B21=1, B22=0 -> [1,0,1,0]
            # W_0: C11=1, C12=0, C21=0, C22=1 -> [1,0,0,1]
            
            U[0,0]=1; U[1,0]=0; U[2,0]=0; U[3,0]=1
            V[0,0]=1; V[1,0]=0; V[2,0]=1; V[3,0]=0
            W[0,0]=1; W[1,0]=0; W[2,0]=0; W[3,0]=1
            
            # r=1 (M2): (A21+A22)B11
            # U_1: [0,0,1,1]
            # V_1: [1,0,0,0]
            # W_1: [0,0,1,0] (C21)
            U[0,1]=0; U[1,1]=0; U[2,1]=1; U[3,1]=1
            V[0,1]=1; V[1,1]=0; V[2,1]=0; V[3,1]=0
            W[0,1]=0; W[1,1]=0; W[2,1]=1; W[3,1]=0
            
            # r=2 (M3): A11(B12-B22)
            # U_2: [1,0,0,0]
            # V_2: [0,1,0,-1]
            # W_2: [0,1,0,0] (C12)
            U[0,2]=1; U[1,2]=0; U[2,2]=0; U[3,2]=0
            V[0,2]=0; V[1,2]=1; V[2,2]=0; V[3,2]=-1
            W[0,2]=0; W[1,2]=1; W[2,2]=0; W[3,2]=0
            
            # r=3 (M4): A22(B21-B11)
            # U_3: [0,0,0,1]
            # V_3: [0,0,1,-1]
            # W_3: [1,0,1,0] (C11, C21)
            U[0,3]=0; U[1,3]=0; U[2,3]=0; U[3,3]=1
            V[0,3]=0; V[1,3]=0; V[2,3]=1; V[3,3]=-1
            W[0,3]=1; W[1,3]=0; W[2,3]=1; W[3,3]=0
            
            # r=4 (M5): (A11+A12)B22
            # U_4: [1,1,0,0]
            # V_4: [0,0,0,1]
            # W_4: [-1,1,0,0] (C11, C12)
            U[0,4]=1; U[1,4]=1; U[2,4]=0; U[3,4]=0
            V[0,4]=0; V[1,4]=0; V[2,4]=0; V[3,4]=1
            W[0,4]=-1; W[1,4]=1; W[2,4]=0; W[3,4]=0
            
            # r=5 (M6): (A21-A11)(B11+B12)
            # U_5: [-1,0,1,0]
            # V_5: [1,1,0,0]
            # W_5: [0,0,1,0] (C21)
            U[0,5]=-1; U[1,5]=0; U[2,5]=1; U[3,5]=0
            V[0,5]=1; V[1,5]=1; V[2,5]=0; V[3,5]=0
            W[0,5]=0; W[1,5]=0; W[2,5]=1; W[3,5]=0
            
            # r=6 (M7): (A12-A22)(B21+B22)
            # U_6: [0,1,0,-1]
            # V_6: [0,0,1,1]
            # W_6: [-1,0,0,1] (C11, C22)
            U[0,6]=0; U[1,6]=1; U[2,6]=0; U[3,6]=-1
            V[0,6]=0; V[1,6]=0; V[2,6]=1; V[3,6]=1
            W[0,6]=-1; W[1,6]=0; W[2,6]=0; W[3,6]=1
            
            return U, V, W, 7

        # Recursive step: Split the largest dimension
        # Priorities: Split m if m>1 and m is even or >2?
        # If m=3, split into 2,1.
        # If p=5, split into 2,3.
        # If n=2, don't split.
        
        # Heuristic: Split m first, then p, then n.
        # If m > 1:
        if m > 1:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1, r1 = get_decomp(n, m1, p)
            U2, V2, W2, r2 = get_decomp(n, m2, p)
            
            # Combine
            U = np.hstack([U1, U2])
            # V needs to be permuted/blocked.
            # V1 is (m1*p, r1). V2 is (m2*p, r2).
            # Total V is (m*p, r1+r2).
            V = np.zeros((m * p, r1 + r2))
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            
            # W is (p*n, r1+r2)
            W = np.hstack([W1, W2])
            
            return U, V, W, r1 + r2

        # If p > 1:
        if p > 1:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1, r1 = get_decomp(n, m, p1)
            U2, V2, W2, r2 = get_decomp(n, m, p2)
            
            U = np.hstack([U1, U2])
            
            # V1 is (m*p1, r1). V2 is (m*p2, r2).
            # Total V is (m*p, r1+r2).
            # V index is j*p + k.
            # For V1, k in 0..p1-1.
            # For V2, k in p1..p-1.
            V = np.zeros((m * p, r1 + r2))
            # Fill V1
            for j in range(m):
                for k in range(p1):
                    V[j*p + k, :r1] = V1[j*p1 + k, :r1]
            # Fill V2
            for j in range(m):
                for k in range(p2):
                    V[j*p + (p1+k), r1:] = V2[j*p2 + k, :r2]
            
            # W1 is (p1*n, r1). W2 is (p2*n, r2).
            # Total W is (p*n, r1+r2).
            # W index is k*n + i.
            W = np.zeros((p * n, r1 + r2))
            W[:p1*n, :r1] = W1
            W[p1*n:, r1:] = W2
            
            return U, V, W, r1 + r2

        # If n > 1:
        if n > 1:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1, r1 = get_decomp(n1, m, p)
            U2, V2, W2, r2 = get_decomp(n2, m, p)
            
            # U1 is (n1*m, r1). U2 is (n2*m, r2).
            # Total U is (n*m, r1+r2).
            # U index is i*m + j.
            U = np.zeros((n * m, r1 + r2))
            for i in range(n1):
                for j in range(m):
                    U[i*m + j, :r1] = U1[i*m1 + j, :r1]
            for i in range(n2):
                for j in range(m):
                    U[(n1+i)*m + j, r1:] = U2[i*m + j, :r2]
            
            V = np.hstack([V1, V2])
            
            # W1 is (p*n1, r1). W2 is (p*n2, r2).
            # Total W is (p*n, r1+r2).
            # W index is k*n + i.
            W = np.zeros((p * n, r1 + r2))
            for k in range(p):
                W[k*n : k*n+n1, :r1] = W1[k*n1 : k*n1+n1, :r1]
                W[k*n+n1 : k*n+n, r1:] = W2[k*n2 : k*n2+n2, :r2]
            
            return U, V, W, r1 + r2

        # Fallback
        rank = n * m * p
        U = np.eye(n * m, rank)
        V = np.eye(m * p, rank)
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    W[k * n + i, idx] = 1
        return U, V, W, rank

    U, V, W, rank = get_decomp(n, m, p)
    
    # Convert to lists of lists for JSON/interface if needed, 
    # but the task says "factor matrices ... with integer/half-integer entries".
    # The seed program returns lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
