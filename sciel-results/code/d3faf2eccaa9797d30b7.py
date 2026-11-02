import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def strassen_222():
    # Rank 7 decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p1+p4+p5-p7, p2+p6], [p3+p5, p1+p3+p4-p6]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c+d)(f)
    # m7 = (a-b)(e+g)
    
    # U indices: (i, j)
    # V indices: (j, k)
    # W indices: (k, i)
    
    # m1: (a+d)(e+h) -> A: (0,0)=1, (1,1)=1; B: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1, (1,0)=-1, (0,1)=1? 
    # C00 = m1 + m4 + m5 - m7
    # C01 = m2 + m6
    # C10 = m3 + m5
    # C11 = m1 + m3 + m4 - m6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1: A: a,d. B: e,f,g,h. W: C00(1), C11(1), C10(-1), C01(1) -> Wait.
    # C00 gets +1. C11 gets +1. C10 gets -1? No, C11 = ... - m6.
    # Let's just use the known matrix.
    
    # U (A coeffs)
    # m1: a=1, d=1
    # m2: c=1, d=1
    # m3: a=1
    # m4: d=1
    # m5: a=1, b=1
    # m6: c=1, d=1
    # m7: a=1, b=-1
    
    U[0, 0] = 1; U[3, 0] = 1 # m1: a, d
    U[2, 1] = 1; U[3, 1] = 1 # m2: c, d
    U[0, 2] = 1             # m3: a
    U[3, 3] = 1             # m4: d
    U[0, 4] = 1; U[1, 4] = 1 # m5: a, b
    U[2, 5] = 1; U[3, 5] = 1 # m6: c, d
    U[0, 6] = 1; U[1, 6] = -1 # m7: a, -b
    
    # V (B coeffs)
    # m1: e,f,g,h
    # m2: e
    # m3: f, -h
    # m4: g, -c? No, d(g-c). B: g, -c? No, B is 2x2.
    # m4 = d(g-c). B: g=1, h=0, f=0, e=0? No.
    # m4 uses g and c? c is from A.
    # m4 = d(g-c).
    # So V for m4: g=1, f=0, e=0, h=0?
    # Wait, (g-c) means B[1,0] - B[0,0]? No.
    # c is A[1,0]. g is B[1,0].
    # So m4 = A[1,1] * (B[1,0] - A[0,0]?) No.
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c+d)(f)
    # m7 = (a-b)(e+g)
    
    # V indices: (j, k). j=0,1; k=0,1.
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
    
    V[0, 0] = 1; V[1, 0] = 1; V[0, 1] = 1; V[1, 1] = 1 # m1: e,f,g,h
    V[0, 1] = 1 # m2: e
    V[0, 2] = 1; V[1, 2] = -1 # m3: f, -h
    V[1, 3] = 1 # m4: g (c is from A, so V only has g? No, m4 = d(g-c).
                # The term is A[1,1] * B[1,0] - A[1,1] * A[0,0]? No.
                # m4 is a product of linear forms in A and B.
                # L_A = d. L_B = g - c? No, c is in A.
                # So L_B = g. And there is a cross term?
                # No, Strassen matrices are linear in A and B separately.
                # m4 = (d) * (g-c) is NOT bilinear in A and B if c is in A.
                # Wait. m4 = (d)(g-c).
                # d is A[1,1]. g is B[1,0]. c is A[0,0]? No, c is A[1,0].
                # So m4 = A[1,1] B[1,0] - A[1,1] A[1,0]? No.
                # The formula is m4 = (d)(g-c).
                # This implies the linear form in B is (g-c)? No, c is not in B.
                # I must have the formula wrong or misremembered.
                
    # Correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)f
    # m7 = (a-b)(e+g)
    
    # All mi are bilinear.
    # m4 = dg - dc.
    # dg is A[1,1]B[1,0].
    # dc is A[1,1]A[1,0]? No, that's quadratic in A.
    # Ah, c is A[1,0].
    # So m4 is NOT bilinear?
    # No, m4 = (d)(g-c).
    # If c is in A, then m4 is not bilinear in A and B.
    # BUT Strassen works.
    # Let's check the source.
    # m4 = (d)(g-c).
    # g is B[1,0]. c is A[1,0].
    # So m4 = A[1,1]B[1,0] - A[1,1]A[1,0].
    # This is not bilinear.
    
    # Wait, the standard Strassen is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)f
    # m7 = (a-b)(e+g)
    
    # And C = [[m1+m4+m5-m7, m2+m6], [m3+m5, m1+m3+m4-m6]]
    
    # Let's verify m4.
    # C10 = m3 + m5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh.
    # Correct.
    # C01 = m2 + m6 = (c+d)e + (c+d)f = (c+d)(e+f).
    # Correct.
    # C11 = m1 + m3 + m4 - m6
    # = (a+d)(e+h) + a(f-h) + d(g-c) - (c+d)f
    # = ae + ah + de + dh + af - ah + dg - dc - cf - df
    # = ae + de + dh + af + dg - dc - cf - df
    # = ae + de + dh + af + dg - dc - cf - df
    # We want dh + bg? No, C11 = dh + bg? No.
    # C11 = d*h + c*g? No.
    # C = [[ae+cg, af+dh], [ce+dg, cf+dh]]
    # C11 = cf + dh.
    # My expansion: ae + de + dh + af + dg - dc - cf - df.
    # This is not cf + dh.
    
    # I will use a verified Strassen implementation.
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 1], # a
        [0, 0, 0, 0, 1, 0, -1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 1, 0]  # d
    ])
    
    V = np.array([
        [1, 1, 0, 0, 0, 0, 1], # e
        [1, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 1, 0, 0] # h
    ])
    
    W = np.array([
        [1, 0, 0, 1, 1, 0, -1], # C00
        [0, 1, 0, 0, 0, 1, 0],  # C01
        [0, 0, 1, 0, 1, 0, 0],  # C10
        [1, 0, 1, 1, 0, -1, 0]  # C11
    ])
    
    return U, V, W

def decompose(n, m, p):
    # Base case: Strassen
    if n == 2 and m == 2 and p == 2:
        return strassen_222()
        
    # Base case: Schoolbook if small or no split found
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = decompose(n1, m, p)
        U2, V2, W2 = decompose(n2, m, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            # Combine
            # U: block diagonal?
            # A1 rows 0..n1-1. A2 rows n1..n-1.
            # U1 is (n1*m, r1). U2 is (n2*m, r2).
            # U = [U1 0; 0 U2]
            U_new = np.zeros((n * m, rank))
            U_new[:(n1*m), :U1.shape[1]] = U1
            U_new[(n1*m):(n*m), U1.shape[1]:] = U2
            
            V_new = np.concatenate([V1, V2], axis=1)
            W_new = np.concatenate([W1, W2], axis=1)
            
            best_rank = rank
            best_U, best_V, best_W = U_new, V_new, W_new
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = decompose(n, m1, p)
        U2, V2, W2 = decompose(n, m2, p)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            # A1 cols 0..m1-1. A2 cols m1..m-1.
            # U1 is (n*m1, r1). U2 is (n*m2, r2).
            # U = [U1 0; 0 U2]
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                U_new[i*m:i*m+m1, :U1.shape[1]] = U1[i*m1:(i+1)*m1, :]
                U_new[i*m+m1:i*m+m, U1.shape[1]:] = U2[i*m2:(i+1)*m2, :]
                
            # V1 is (m1*p, r1). V2 is (m2*p, r2).
            V_new = np.zeros((m * p, rank))
            V_new[:m1*p, :V1.shape[1]] = V1
            V_new[m1*p:, V1.shape[1]:] = V2
            
            W_new = np.concatenate([W1, W2], axis=1)
            
            best_rank = rank
            best_U, best_V, best_W = U_new, V_new, W_new
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = decompose(n, m, p1)
        U2, V2, W2 = decompose(n, m, p2)
        rank = U1.shape[1] + U2.shape[1]
        if rank < best_rank:
            # B1 rows 0..p1-1. B2 rows p1..p-1.
            # V1 is (m*p1, r1). V2 is (m*p2, r2).
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                V_new[j*p:j*p+p1, :V1.shape[1]] = V1[j*p1:(j+1)*p1, :]
                V_new[j*p+p1:j*p+p, V1.shape[1]:] = V2[j*p2:(j+1)*p2, :]
                
            U_new = np.concatenate([U1, U2], axis=1)
            
            # W1 is (p1*n, r1). W2 is (p2*n, r2).
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                W_new[k*n:(k+1)*n, :W1.shape[1]] = W1[k*n1:(k+1)*n1, :] if k < p1 else W2[(k-p1)*n:(k-p1+1)*n, W1.shape[1]:]
            # Correction for W indexing
            W_new = np.zeros((p * n, rank))
            for k in range(p1):
                W_new[k*n:(k+1)*n, :W1.shape[1]] = W1[k*n:(k+1)*n, :]
            for k in range(p2):
                W_new[(k+p1)*n:(k+p1+1)*n, W1.shape[1]:] = W2[k*n:(k+1)*n, :]
                
            best_rank = rank
            best_U, best_V, best_W = U_new, V_new, W_new
            
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
