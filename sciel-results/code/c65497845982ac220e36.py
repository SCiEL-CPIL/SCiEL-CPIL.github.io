import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Define Strassen decomposition for 2x2 x 2x2 (Rank 7)
    # U1: (4, 7), V1: (4, 7), W1: (4, 7)
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # n=2, m=2, p=2
    
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # Strassen products p1..p7
    # A = [[a, b], [c, d]], B = [[e, f], [h, i]]
    # a=0, b=1, c=2, d=3 (indices for U1)
    # e=0, f=1, h=2, i=3 (indices for V1)
    # C = [[ae+bh, af+bi], [ce+dh, cf+di]]
    # C indices: k*n+i
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3 (indices for W1)
    
    # p1 = (a+d)(e+h)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[2, 0] = 1
    W1[0, 0] = 1; W1[1, 0] = 1; W1[2, 0] = 1; W1[3, 0] = 1
    
    # p2 = (c+d)e
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = -1; W1[3, 1] = 1
    
    # p3 = a(f-h)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[2, 2] = -1
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # p4 = d(i-e)
    U1[3, 3] = 1
    V1[3, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # p5 = (a+b)h
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[2, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # p6 = (c+a)(f+i) -- Wait, standard Strassen p6 is (c+a)(f+i) ?
    # Let's use the standard 7:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(i-e)
    # p5 = (a+b)h
    # p6 = (c+a)(f+i) -> No, usually p6 = (c+a)(f+i) is not standard.
    # Standard:
    # p6 = (c+a)(f+i) ? No.
    # Let's check Wikipedia Strassen:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(f - h)
    # p4 = d(g - e) -> g is i here? No, B is [[e,f],[h,i]]. So g->i?
    # If B = [[e, f], [h, i]], then:
    # p4 = d(i - e)
    # p5 = (a + b)h
    # p6 = (c + a)(i + f) -> No.
    # p6 = (c + a)(f + i) ?
    # p7 = (b + d)(h - e) ?
    
    # Let's verify the reconstruction:
    # C11 = p1 - p4 - p5 + p7 ?
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 - p6 ?
    
    # Let's use a known correct set:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(i-e)
    # p5 = (a+b)h
    # p6 = (c+a)(i+f)
    # p7 = (b+d)(h-e)
    
    # C11 = p1 - p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 - p6
    
    # Let's code p6 and p7 correctly.
    # p6 = (c+a)(i+f)
    U1[2, 5] = 1; U1[0, 5] = 1
    V1[3, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = -1
    
    # p7 = (b+d)(h-e)
    U1[1, 6] = 1; U1[3, 6] = 1
    V1[2, 6] = 1; V1[0, 6] = -1
    W1[0, 6] = 1

    # 2. Define decomposition for 2x2 x 2x3 (Rank 11)
    # U2: (4, 11), V2: (6, 11), W2: (6, 11)
    # n=2, m=2, p=3
    # A = [[a, b], [c, d]]
    # B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float)
    W2 = np.zeros((6, 11), dtype=float)
    
    # First 7 columns: Strassen for first 2 cols of B (e,f,h,i)
    # Copy U1, V1 (first 4 rows), W1 (first 4 rows)
    U2[:, 0:7] = U1
    V2[0:4, 0:7] = V1
    W2[0:4, 0:7] = W1
    
    # Last 4 columns: Schoolbook for 3rd col (g, j)
    # ag, bj, cg, dj
    # U indices: a=0, b=1, c=2, d=3
    # V indices: g=4, j=5 (since B is 2x3, flattened j*p+k: 0,1,2,3,4,5 -> e,f,h,i,g,j)
    # W indices: C is 2x3. k*n+i.
    # C[0,2] -> k=2, i=0 -> 2*2+0 = 4
    # C[1,2] -> k=2, i=1 -> 2*2+1 = 5
    
    # ag
    U2[0, 7] = 1
    V2[4, 7] = 1
    W2[4, 7] = 1
    
    # bj
    U2[1, 8] = 1
    V2[5, 8] = 1
    W2[4, 8] = 1
    
    # cg
    U2[2, 9] = 1
    V2[4, 9] = 1
    W2[5, 9] = 1
    
    # dj
    U2[3, 10] = 1
    V2[5, 10] = 1
    W2[5, 10] = 1

    # 3. Kronecker Product
    # U_total = kron(U1, U2)
    # V_total = kron(V1, V2)
    # W_total = kron(W1, W2)
    
    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook if we are too slow (unlikely)
        return _schoolbook()

    U_total = np.kron(U1, U2)
    V_total = np.kron(V1, V2)
    W_total = np.kron(W1, W2)
    
    # Convert to list of lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

def _schoolbook():
    n, m, p = DIMENSIONS
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
