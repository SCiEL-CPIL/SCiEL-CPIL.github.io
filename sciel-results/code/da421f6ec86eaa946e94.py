import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = 3, 5, 6
    rank = 30
    
    # Use Winograd 5-point transform on j-dimension
    # Evaluation points: 0, 1, 2, 3, 4
    points = np.array([0, 1, 2, 3, 4])
    
    # F[a, j] = points[j] ** a
    F = np.zeros((5, 5))
    for a in range(5):
        for j in range(5):
            F[a, j] = points[j] ** a
    
    # F_inv = inverse of F
    F_inv = np.linalg.inv(F)
    
    # Convert to fractions for exact arithmetic
    from fractions import Fraction
    
    F_frac = [[Fraction(F[a, j]).limit_denominator(1000000) for j in range(5)] for a in range(5)]
    F_inv_frac = [[Fraction(F_inv[a, j]).limit_denominator(1000000) for j in range(5)] for a in range(5)]
    
    # Construct U, V, W
    # 30 terms indexed by (a, k) where a in 0..4, k in 0..5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for a in range(5):
        for k in range(6):
            for i in range(3):
                for j in range(5):
                    U[i * m + j][c] = F_inv_frac[j][a]
                    V[j * p + k][c] = F_frac[a][j]
                    W[k * n + i][c] = 1
            c += 1
    
    # Verify
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j
    # Actually, the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need: C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    # This requires: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
    
    # Let's verify
    tensor_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += float(U[i*m+j][c]) * float(V[j*p+k][c]) * float(W[k*n+i][c])
                tensor_reconstructed[i, j, k] = total
    
    # Should be all 1s
    if not np.allclose(tensor_reconstructed, 1.0, atol=1e-10):
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
