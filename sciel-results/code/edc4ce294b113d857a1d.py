import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Rank-11 decomposition: Strassen (rank 7) for 2x2x2 block + schoolbook (rank 4) for remaining column
    # U shape: (4, 11), V shape: (6, 11), W shape: (6, 11)
    # U indices: i*m+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: j*p+k -> (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # W indices: k*n+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5

    R = 11
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]

    # Strassen terms (rank 7) for columns k=0,1
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # Term 2: M3
    U[0][2] = 1
    V[3][2] = 1; V[0][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # Term 3: M4
    U[3][3] = 1
    V[1][3] = 1; V[4][3] = -1
    W[1][3] = 1; W[3][3] = 1
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    # Schoolbook terms (rank 4) for column k=2
    # C00 += A00*B02
    # C01 += A01*B02  
    # C10 += A10*B12
    # C11 += A11*B12
    
    # Term 7: A00*B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 8: A01*B02
    U[1][8] = 1
    V[2][8] = 1
    W[5][8] = 1
    
    # Term 9: A10*B12
    U[2][9] = 1
    V[5][9] = 1
    W[2][9] = 1
    
    # Term 10: A11*B12
    U[3][10] = 1
    V[5][10] = 1
    W[3][10] = 1

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(R):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to rank 12
        R = 12
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1

    return {"U": U, "V": V, "W": W}
