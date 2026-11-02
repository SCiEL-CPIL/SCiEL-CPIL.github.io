import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # Appears in: C[0,k0] with coeff +1, C[1,k1] with coeff +1
        U[0*m+0, r] = 1
        U[1*m+1, r] = 1
        V[0*p+k0, r] = 1
        V[1*p+k1, r] = 1
        W[k0*n+0, r] = 1
        W[k1*n+1, r] = 1
        r += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # Appears in: C[1,k0] with coeff +1, C[1,k1] with coeff -1
        U[1*m+0, r] = 1
        U[1*m+1, r] = 1
        V[0*p+k0, r] = 1
        W[k0*n+1, r] = 1
        W[k1*n+1, r] = -1
        r += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        # Appears in: C[0,k1] with coeff +1, C[1,k1] with coeff +1
        U[0*m+0, r] = 1
        V[0*p+k1, r] = 1
        V[1*p+k1, r] = -1
        W[k1*n+0, r] = 1
        W[k1*n+1, r] = 1
        r += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        # Appears in: C[0,k0] with coeff +1, C[1,k0] with coeff +1
        U[1*m+1, r] = 1
        V[1*p+k0, r] = 1
        V[0*p+k0, r] = -1
        W[k0*n+0, r] = 1
        W[k0*n+1, r] = 1
        r += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # Appears in: C[0,k0] with coeff -1, C[0,k1] with coeff +1
        U[0*m+0, r] = 1
        U[0*m+1, r] = 1
        V[1*p+k1, r] = 1
        W[k0*n+0, r] = -1
        W[k1*n+0, r] = 1
        r += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # Appears in: C[1,k1] with coeff +1
        U[1*m+0, r] = 1
        U[0*m+0, r] = -1
        V[0*p+k0, r] = 1
        V[0*p+k1, r] = 1
        W[k1*n+1, r] = 1
        r += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        # Appears in: C[0,k0] with coeff +1
        U[0*m+1, r] = 1
        U[1*m+1, r] = -1
        V[1*p+k0, r] = 1
        V[1*p+k1, r] = 1
        W[k0*n+0, r] = 1
        r += 1
        
        # 4 schoolbook for A[:,2] @ B[2, k0:k1+1]
        U[0*m+2, r] = 1; V[2*p+k0, r] = 1; W[k0*n+0, r] = 1; r += 1
        U[0*m+2, r] = 1; V[2*p+k1, r] = 1; W[k1*n+0, r] = 1; r += 1
        U[1*m+2, r] = 1; V[2*p+k0, r] = 1; W[k0*n+1, r] = 1; r += 1
        U[1*m+2, r] = 1; V[2*p+k1, r] = 1; W[k1*n+1, r] = 1; r += 1
    
    assert r == rank
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                assert abs(val - 1.0) < 1e-10, f"Error at ({i},{j},{k}): {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
