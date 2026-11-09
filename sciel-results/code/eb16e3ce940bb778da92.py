"""Matrix multiplication 4x5x5: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Partition into 2x2x2 Strassen blocks where possible, schoolbook otherwise.
    Total rank = 92 (8 Strassen blocks * 7 + 36 schoolbook terms).
    """
    deadline = time.time() + 140.0
    n, m, p = 4, 5, 5
    
    # Define partitions
    I_groups = [[0, 1], [2, 3]]
    J_groups = [[0, 1], [2, 3], [4]]
    K_groups = [[0, 1], [2, 3], [4]]
    
    # Pre-allocate with max possible rank
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    for Ig in I_groups:
        for Jg in J_groups:
            for Kg in K_groups:
                if len(Ig) == 2 and len(Jg) == 2 and len(Kg) == 2:
                    i0, i1 = Ig[0], Ig[1]
                    j0, j1 = Jg[0], Jg[1]
                    k0, k1 = Kg[0], Kg[1]
                    
                    # Strassen 7-term decomposition
                    # m1 = (a11+a22)(b11+b22)
                    U[i0*m+j0][col] += 1; U[i1*m+j1][col] += 1
                    V[j0*p+k0][col] += 1; V[j1*p+k1][col] += 1
                    W[k0*n+i0][col] += 1; W[k1*n+i1][col] += 1
                    col += 1
                    
                    # m2 = (a21+a22)b11
                    U[i1*m+j0][col] += 1; U[i1*m+j1][col] += 1
                    V[j0*p+k0][col] += 1
                    W[k0*n+i1][col] += 1
                    col += 1
                    
                    # m3 = a11(b12-b22)
                    U[i0*m+j0][col] += 1
                    V[j0*p+k1][col] += 1; V[j1*p+k1][col] -= 1
                    W[k1*n+i0][col] += 1
                    col += 1
                    
                    # m4 = a22(b21-b11)
                    U[i1*m+j1][col] += 1
                    V[j1*p+k0][col] += 1; V[j0*p+k0][col] -= 1
                    W[k0*n+i1][col] += 1
                    col += 1
                    
                    # m5 = (a11+a12)b22
                    U[i0*m+j0][col] += 1; U[i0*m+j1][col] += 1
                    V[j1*p+k1][col] += 1
                    W[k1*n+i0][col] += 1
                    col += 1
                    
                    # m6 = (a21-a11)(b11+b12)
                    U[i1*m+j0][col] += 1; U[i0*m+j0][col] -= 1
                    V[j0*p+k0][col] += 1; V[j0*p+k1][col] += 1
                    W[k0*n+i1][col] += 1; W[k1*n+i1][col] += 1
                    col += 1
                    
                    # m7 = (a12-a22)(b21+b22)
                    U[i0*m+j1][col] += 1; U[i1*m+j1][col] -= 1
                    V[j1*p+k0][col] += 1; V[j1*p+k1][col] += 1
                    W[k0*n+i0][col] += 1; W[k1*n+i0][col] += 1
                    col += 1
                else:
                    # Schoolbook terms
                    for i in Ig:
                        for j in Jg:
                            for k in Kg:
                                U[i*m+j][col] += 1
                                V[j*p+k][col] += 1
                                W[k*n+i][col] += 1
                                col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    rank = col
    
    # Verify correctness
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        import random
        random.seed(42)
        ok = True
        for _ in range(2000):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            kp = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            
            expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
            actual = float(np.sum(U_arr[i*m+j] * V_arr[jp*p+k] * W_arr[kp*n+ip]))
            if abs(actual - expected) > 1e-9:
                ok = False
                break
        
        if not ok:
            # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
