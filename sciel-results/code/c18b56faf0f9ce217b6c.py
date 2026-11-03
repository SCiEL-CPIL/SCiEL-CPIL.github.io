import numpy as np

def construct():
    n, m, p = 2, 4, 8
    
    # Factor 1: Strassen 2×2·2×2, rank 7
    n1, m1, p1, r1 = 2, 2, 2, 7
    U1 = np.zeros((n1*m1, r1))
    V1 = np.zeros((m1*p1, r1))
    W1 = np.zeros((p1*n1, r1))
    
    # Strassen: C = A*B for 2×2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Index: U1[i*m1+j, r] for A[i,j], V1[j*p1+k, r] for B[j,k], W1[k*n1+i, r] for C[i,k]
    
    # M1 = (A00+A11)(B00+B11)
    U1[0,0]=1; U1[3,0]=1
    V1[0,0]=1; V1[3,0]=1
    W1[0,0]=1; W1[3,0]=1  # C[0,0]=W1[0*2+0], C[1,1]=W1[1*2+1]
    
    # M2 = (A10+A11)B00
    U1[2,1]=1; U1[3,1]=1
    V1[0,1]=1
    W1[2,1]=1; W1[3,1]=-1  # C[1,0]=W1[0*2+1], C[1,1]=W1[1*2+1]
    
    # M3 = A00(B10-B11)
    U1[0,2]=1
    V1[2,2]=1; V1[3,2]=-1
    W1[1,2]=1; W1[3,2]=1  # C[0,1]=W1[1*2+0], C[1,1]=W1[1*2+1]
    
    # M4 = A11(B01-B10)
    U1[3,3]=1
    V1[1,3]=1; V1[2,3]=-1
    W1[0,3]=1; W1[2,3]=1  # C[0,0]=W1[0*2+0], C[1,0]=W1[0*2+1]
    
    # M5 = (A00+A01)B11
    U1[0,4]=1; U1[1,4]=1
    V1[3,4]=1
    W1[0,4]=-1; W1[1,4]=1  # C[0,0]=W1[0*2+0], C[0,1]=W1[1*2+0]
    
    # M6 = (A10-A00)(B00+B01)
    U1[2,5]=1; U1[0,5]=-1
    V1[0,5]=1; V1[1,5]=1
    W1[3,5]=1  # C[1,1]=W1[1*2+1]
    
    # M7 = (A01-A11)(B10+B11)
    U1[1,6]=1; U1[3,6]=-1
    V1[2,6]=1; V1[3,6]=1
    W1[0,6]=1  # C[0,0]=W1[0*2+0]
    
    # Factor 2: Schoolbook 1×2·2×4, rank 8
    n2, m2, p2, r2 = 1, 2, 4, 8
    U2 = np.zeros((n2*m2, r2))
    V2 = np.zeros((m2*p2, r2))
    W2 = np.zeros((p2*n2, r2))
    
    for j in range(m2):
        for k in range(p2):
            r = j*p2 + k
            U2[0*m2+j, r] = 1
            V2[j*p2+k, r] = 1
            W2[k*n2+0, r] = 1
    
    # Kronecker product
    U = np.zeros((n1*n2, r1*r2))
    V = np.zeros((m1*m2, r1*r2))
    W = np.zeros((p1*p2, r1*r2))
    
    for r1i in range(r1):
        for r2i in range(r2):
            r = r1i*r2 + r2i
            U[:, r] = np.kron(U1[:, r1i], U2[:, r2i])
            V[:, r] = np.kron(V1[:, r1i], V2[:, r2i])
            W[:, r] = np.kron(W1[:, r1i], W2[:, r2i])
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                assert abs(val - 1.0) < 1e-8, f"Failed at ({i},{j},{k}): {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
