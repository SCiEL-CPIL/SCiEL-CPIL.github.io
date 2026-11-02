import time
import numpy as np

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS

    # Build target tensor for validation
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_tensor[u, v, w] = 1.0

    def to_half_int_list(matrix):
        scaled = np.rint(matrix * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]

    def validate(U, V, W):
        """Check if U, V, W decompose the tensor exactly."""
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, target_tensor, atol=1e-8)

    # Construct rank-17 decomposition using Strassen for 2x2 subproblem + schoolbook
    # A is 2x3, B is 3x3, C is 2x3
    # 
    # Strategy: Use Strassen for the "core" 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    # Then add schoolbook terms for remaining parts.
    #
    # Standard Strassen for 2x2 matrix multiplication:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11+b12)
    # M3 = (a11-a22)(b21+b22)
    # M4 = a11(b12-b22)
    # M5 = a22(b21-b11)
    # M6 = (a11+a12)(b22-b12)
    # M7 = (a21+a22)(b11-b21)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M4 - M6 - M7
    # C21 = M2 + M5 - M6 - M7
    # C22 = M1 + M3 - M4 + M6
    #
    # For our case: A[i,j] where i in {0,1}, j in {0,1}; B[j,k] where j in {0,1}, k in {0,1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Mapping: a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
    #           b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    #           C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    rank = 17
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # Contributes: +1 to C[0,0] (C11), +1 to C[1,1] (C22)
    U[0*3+0, col] = 1.0  # A[0,0]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[0*3+0, col] = 1.0  # B[0,0]
    V[1*3+1, col] = 1.0  # B[1,1]
    W[0*2+0, col] = 1.0  # C[0,0]
    W[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M2 = (A[1,0]+A[1,1])*(B[0,0]+B[0,1])
    # Contributes: +1 to C[0,1] (C12), +1 to C[1,0] (C21)
    U[1*3+0, col] = 1.0  # A[1,0]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[0*3+0, col] = 1.0  # B[0,0]
    V[0*3+1, col] = 1.0  # B[0,1]
    W[0*2+1, col] = 1.0  # C[0,1]
    W[0*2+1, col] = 1.0  # C[1,0] -- wait, W index is k*n+i
    # C[1,0]: k=0, i=1, so W[0*2+1] = W[1]
    # C[0,1]: k=1, i=0, so W[1*2+0] = W[2]
    W[1*2+0, col] = 1.0  # C[0,1]
    W[0*2+1, col] = 1.0  # C[1,0]
    col += 1
    
    # M3 = (A[0,0]-A[1,1])*(B[1,0]+B[1,1])
    # Contributes: +1 to C[1,1] (C22)
    U[0*3+0, col] = 1.0  # A[0,0]
    U[1*3+1, col] = -1.0  # A[1,1]
    V[1*3+0, col] = 1.0  # B[1,0]
    V[1*3+1, col] = 1.0  # B[1,1]
    W[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M4 = A[0,0]*(B[0,1]-B[1,1])
    # Contributes: +1 to C[0,0] (C11), +1 to C[0,1] (C12), -1 to C[1,1] (C22)
    U[0*3+0, col] = 1.0  # A[0,0]
    V[0*3+1, col] = 1.0  # B[0,1]
    V[1*3+1, col] = -1.0  # B[1,1]
    W[0*2+0, col] = 1.0  # C[0,0]
    W[1*2+0, col] = 1.0  # C[0,1]
    W[1*2+1, col] = -1.0  # C[1,1]
    col += 1
    
    # M5 = A[1,1]*(B[1,0]-B[0,0])
    # Contributes: -1 to C[0,0] (C11), +1 to C[1,0] (C21)
    U[1*3+1, col] = 1.0  # A[1,1]
    V[1*3+0, col] = 1.0  # B[1,0]
    V[0*3+0, col] = -1.0  # B[0,0]
    W[0*2+0, col] = -1.0  # C[0,0]
    W[0*2+1, col] = 1.0  # C[1,0]
    col += 1
    
    # M6 = (A[0,0]+A[0,1])*(B[1,1]-B[0,1])
    # Contributes: -1 to C[0,1] (C12), -1 to C[1,0] (C21), +1 to C[1,1] (C22)
    U[0*3+0, col] = 1.0  # A[0,0]
    U[0*3+1, col] = 1.0  # A[0,1]
    V[1*3+1, col] = 1.0  # B[1,1]
    V[0*3+1, col] = -1.0  # B[0,1]
    W[1*2+0, col] = -1.0  # C[0,1]
    W[0*2+1, col] = -1.0  # C[1,0]
    W[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M7 = (A[1,0]+A[1,1])*(B[0,0]-B[1,0])
    # Contributes: +1 to C[0,0] (C11), -1 to C[0,1] (C12), -1 to C[1,0] (C21)
    U[1*3+0, col] = 1.0  # A[1,0]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[0*3+0, col] = 1.0  # B[0,0]
    V[1*3+0, col] = -1.0  # B[1,0]
    W[0*2+0, col] = 1.0  # C[0,0]
    W[1*2+0, col] = -1.0  # C[0,1]
    W[0*2+1, col] = -1.0  # C[1,0]
    col += 1
    
    # Now add schoolbook terms for the remaining contributions:
    # C[0,0]: A[0,2]*B[2,0]
    # C[0,1]: A[0,2]*B[2,1]
    # C[0,2]: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0]: A[1,2]*B[2,0]
    # C[1,1]: A[1,2]*B[2,1]
    # C[1,2]: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Term 8: A[0,2]*B[2,0] -> C[0,0]
    U[0*3+2, col] = 1.0  # A[0,2]
    V[2*3+0, col] = 1.0  # B[2,0]
    W[0*2+0, col] = 1.0  # C[0,0]
    col += 1
    
    # Term 9: A[0,2]*B[2,1] -> C[0,1]
    U[0*3+2, col] = 1.0  # A[0,2]
    V[2*3+1, col] = 1.0  # B[2,1]
    W[1*2+0, col] = 1.0  # C[0,1]
    col += 1
    
    # Term 10: A[0,0]*B[0,2] -> C[0,2]
    U[0*3+0, col] = 1.0  # A[0,0]
    V[0*3+2, col] = 1.0  # B[0,2]
    W[2*2+0, col] = 1.0  # C[0,2]
    col += 1
    
    # Term 11: A[0,1]*B[1,2] -> C[0,2]
    U[0*3+1, col] = 1.0  # A[0,1]
    V[1*3+2, col] = 1.0  # B[1,2]
    W[2*2+0, col] = 1.0  # C[0,2]
    col += 1
    
    # Term 12: A[0,2]*B[2,2] -> C[0,2]
    U[0*3+2, col] = 1.0  # A[0,2]
    V[2*3+2, col] = 1.0  # B[2,2]
    W[2*2+0, col] = 1.0  # C[0,2]
    col += 1
    
    # Term 13: A[1,2]*B[2,0] -> C[1,0]
    U[1*3+2, col] = 1.0  # A[1,2]
    V[2*3+0, col] = 1.0  # B[2,0]
    W[0*2+1, col] = 1.0  # C[1,0]
    col += 1
    
    # Term 14: A[1,2]*B[2,1] -> C[1,1]
    U[1*3+2, col] = 1.0  # A[1,2]
    V[2*3+1, col] = 1.0  # B[2,1]
    W[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # Term 15: A[1,0]*B[0,2] -> C[1,2]
    U[1*3+0, col] = 1.0  # A[1,0]
    V[0*3+2, col] = 1.0  # B[0,2]
    W[2*2+1, col] = 1.0  # C[1,2]
    col += 1
    
    # Term 16: A[1,1]*B[1,2] -> C[1,2]
    U[1*3+1, col] = 1.0  # A[1,1]
    V[1*3+2, col] = 1.0  # B[1,2]
    W[2*2+1, col] = 1.0  # C[1,2]
    col += 1
    
    # Term 17: A[1,2]*B[2,2] -> C[1,2]
    U[1*3+2, col] = 1.0  # A[1,2]
    V[2*3+2, col] = 1.0  # B[2,2]
    W[2*2+1, col] = 1.0  # C[1,2]
    col += 1
    
    # Verify the decomposition
    if validate(U, V, W):
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    
    # Fallback to schoolbook rank 18 if verification fails
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    return {"U": to_half_int_list(U_sb), "V": to_half_int_list(V_sb), "W": to_half_int_list(W_sb)}
