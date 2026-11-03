"""Rank-49 exact tensor decomposition for 4x4 matrix multiplication via Kronecker product of Strassen."""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _build_strassen_2x2():
    """Return (U2, V2, W2) for 2x2 Strassen with correct index convention.
    
    Convention: U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    Sum over r should give 1 for all i,j,k in {0,1}.
    """
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # m1 = (a+d)(e+h)  -> contributes to C00, C01, C10, C11
    # m2 = (c+d)e      -> contributes to C10
    # m3 = a(f-h)      -> contributes to C01
    # m4 = d(g-e)      -> contributes to C11
    # m5 = (a+b)h      -> contributes to C01, C11
    # m6 = (c-a)(e+f)  -> contributes to C10
    # m7 = (b-d)(g+h)  -> contributes to C00, C11
    
    # U2[i*2+j, r]: coefficient for A[i,j] in product r
    # V2[j*2+k, r]: coefficient for B[j,k] in product r
    # W2[k*2+i, r]: coefficient for C[k,i] in product r (note: trace convention W[k*n+i,r])
    
    # m1 = (a+d)(e+h): A[0,0]+A[1,1] times B[0,0]+B[1,1]
    # C00 += m1, C01 -= m1, C10 += m1, C11 -= m1... wait let me use standard Strassen
    
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-a)*(e+f)
    # p7 = (b-d)*(g+h)
    
    # C[0,0] = p1 - p4 + p5 - p7
    # C[0,1] = p3 + p4
    # C[1,0] = p2 + p7
    # C[1,1] = p1 - p3 + p6 - p7
    
    # U2[i*2+j, r] for A[i,j]
    # V2[j*2+k, r] for B[j,k]  
    # W2[k*2+i, r] for C[k,i] -- this is the trace convention
    
    # r=0: p1 = (a+d)(e+h)
    U2[0, 0] = 1  # a = A[0,0]
    U2[3, 0] = 1  # d = A[1,1]
    V2[0, 0] = 1  # e = B[0,0]
    V2[3, 0] = 1  # h = B[1,1]
    # C[0,0] += p1, C[1,1] += p1, C[0,1] -= p1, C[1,0] -= p1... 
    # Actually from the formulas:
    # C[0,0] = p1 - p4 + p5 - p7  => W2[0*2+0, 0] = W2[0,0] = +1
    # C[0,1] = p3 + p4            => W2[1*2+0, 0] = W2[2,0] = 0 (p1 not in C[0,1])
    # C[1,0] = p2 + p7            => W2[0*2+1, 0] = W2[1,0] = 0 (p1 not in C[1,0])  
    # C[1,1] = p1 - p3 + p6 - p7  => W2[1*2+1, 0] = W2[3,0] = +1
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # r=1: p2 = (c+d)*e
    U2[2, 1] = 1  # c = A[1,0]
    U2[3, 1] = 1  # d = A[1,1]
    V2[0, 1] = 1  # e = B[0,0]
    # C[1,0] = p2 + p7 => W2[0*2+1, 1] = W2[1,1] = +1
    W2[1, 1] = 1
    
    # r=2: p3 = a*(f-h)
    U2[0, 2] = 1  # a = A[0,0]
    V2[1, 2] = 1  # f = B[0,1]
    V2[3, 2] = -1 # h = B[1,1]
    # C[0,1] = p3 + p4 => W2[1*2+0, 2] = W2[2,2] = +1
    # C[1,1] = p1 - p3 + p6 - p7 => W2[1*2+1, 2] = W2[3,2] = -1
    W2[2, 2] = 1
    W2[3, 2] = -1
    
    # r=3: p4 = d*(g-e)
    U2[3, 3] = 1  # d = A[1,1]
    V2[2, 3] = 1  # g = B[1,0]
    V2[0, 3] = -1 # e = B[0,0]
    # C[0,0] = p1 - p4 + p5 - p7 => W2[0*2+0, 3] = W2[0,3] = -1
    # C[0,1] = p3 + p4 => W2[1*2+0, 3] = W2[2,3] = +1
    W2[0, 3] = -1
    W2[2, 3] = 1
    
    # r=4: p5 = (a+b)*h
    U2[0, 4] = 1  # a = A[0,0]
    U2[1, 4] = 1  # b = A[0,1]
    V2[3, 4] = 1  # h = B[1,1]
    # C[0,0] = p1 - p4 + p5 - p7 => W2[0*2+0, 4] = W2[0,4] = +1
    # C[1,1] = p1 - p3 + p6 - p7 => W2[1*2+1, 4] = W2[3,4] = 0 (p5 not in C[1,1])
    # Wait, let me recheck. C[1,1] = p1 - p3 + p6 - p7, no p5.
    # C[0,0] has +p5, so W2[0,4] = +1
    W2[0, 4] = 1
    
    # r=5: p6 = (c-a)*(e+f)
    U2[2, 5] = 1  # c = A[1,0]
    U2[0, 5] = -1 # a = A[0,0]
    V2[0, 5] = 1  # e = B[0,0]
    V2[1, 5] = 1  # f = B[0,1]
    # C[1,1] = p1 - p3 + p6 - p7 => W2[1*2+1, 5] = W2[3,5] = +1
    W2[3, 5] = 1
    
    # r=6: p7 = (b-d)*(g+h)
    U2[1, 6] = 1  # b = A[0,1]
    U2[3, 6] = -1 # d = A[1,1]
    V2[2, 6] = 1  # g = B[1,0]
    V2[3, 6] = 1  # h = B[1,1]
    # C[0,0] = p1 - p4 + p5 - p7 => W2[0*2+0, 6] = W2[0,6] = -1
    # C[1,0] = p2 + p7 => W2[0*2+1, 6] = W2[1,6] = +1
    # C[1,1] = p1 - p3 + p6 - p7 => W2[1*2+1, 6] = W2[3,6] = -1
    W2[0, 6] = -1
    W2[1, 6] = 1
    W2[3, 6] = -1
    
    return U2, V2, W2

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*n+j, r] * V[j*m+k, r] * W[k*p+i, r] = 1 for all i,j,k."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*n+j, :] * V[j*m+k, :] * W[k*p+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    return max_err

def _fallback_schoolbook(n, m, p):
    """Return rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Build 2x2 Strassen
    U2, V2, W2 = _build_strassen_2x2()
    
    # Verify 2x2 Strassen
    err2 = _verify_decomposition(U2, V2, W2, 2, 2, 2)
    
    if err2 >= 1e-10:
        return _fallback_schoolbook(n, m, p)
    
    # Build 4x4 via Kronecker product
    # Index decomposition: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # U4 row index: i*4 + j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # V4 row index: j*4 + k = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2
    # W4 row index: k*4 + i = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2
    # Column: r = r1*7 + r2
    
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        u1 = U2[i1*2+j1, r1]
                        v1 = V2[j1*2+k1, r1]
                        w1 = W2[k1*2+i1, r1]
                        if u1 == 0 or v1 == 0 or w1 == 0:
                            continue
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u2 = U2[i2*2+j2, r2]
                                    v2 = V2[j2*2+k2, r2]
                                    w2 = W2[k2*2+i2, r2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    row_U = 8*i1 + 4*i2 + 2*j1 + j2
                                    row_V = 8*j1 + 4*j2 + 2*k1 + k2
                                    row_W = 8*k1 + 4*k2 + 2*i1 + i2
                                    U4[row_U, col] += u1 * u2
                                    V4[row_V, col] += v1 * v2
                                    W4[row_W, col] += w1 * w2
    
    # Verify 4x4
    err4 = _verify_decomposition(U4, V4, W4, 4, 4, 4)
    
    if err4 < 1e-10:
        U_list = U4.tolist()
        V_list = V4.tolist()
        W_list = W4.tolist()
        
        # Round to integers/half-integers
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val)
                    if abs(val - rounded) < 1e-10:
                        mat[i][j] = rounded
                    else:
                        # Try half-integer
                        half = round(val * 2) / 2
                        if abs(val - half) < 1e-10:
                            mat[i][j] = half
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return _fallback_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
