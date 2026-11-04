import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mappings:
    # U: i*m + j. i in {0,1}, j in {0,1,2}
    # u00=0, u01=1, u02=2, u10=3, u11=4, u12=5
    # V: j*p + k. j in {0,1,2}, k in {0,1,2}
    # v00=0, v01=1, v02=2, v10=3, v11=4, v12=5, v20=6, v21=7, v22=8
    # W: k*n + i. k in {0,1,2}, i in {0,1}
    # w00=0, w01=1, w10=2, w11=3, w20=4, w21=5
    
    # Part 1: 2x2x3 (j in {0,1})
    # Strassen for k in {0,1}
    
    # Col 0: M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2 = (A10 + A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4 = A11(B10 - B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Col 4: M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Col 5: M6 = (A10 - A00)(B00 + B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Standard multiplications for k=2
    # Col 7: M8 = A00 B02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # Col 8: M9 = A01 B12
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # Col 9: M10 = A10 B02
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Col 10: M11 = A11 B12
    U[4, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    # Part 2: 2x1x3 (j=2)
    # Col 11: M12 = A02 B20
    U[2, 11] = 1
    V[6, 11] = 1
    W[0, 11] = 1
    
    # Col 12: M13 = A02 B21
    U[2, 12] = 1
    V[7, 12] = 1
    W[2, 12] = 1
    
    # Col 13: M14 = A02 B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # Col 14: M15 = A12 B20
    U[5, 14] = 1
    V[6, 14] = 1
    W[1, 14] = 1
    
    # Col 15: M16 = A12 B21
    U[5, 15] = 1
    V[7, 15] = 1
    W[3, 15] = 1
    
    # Col 16: M17 = A12 B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
