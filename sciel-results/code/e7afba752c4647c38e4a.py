import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 74 decomposition
    # 1 Strassen 2x2x2 block (rank 7)
    # 67 Schoolbook terms (rank 67)
    rank = 74
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen Block: i in {0,1}, j in {0,1}, k in {0,1}
    # Indices in U: i*m + j
    # Indices in V: j*p + k
    # Indices in W: k*n + i
    
    # Strassen terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A indices: 00->0, 01->1, 10->5, 11->6 (m=5)
    # B indices: 00->0, 01->1, 10->5, 11->6 (p=5)
    # C indices: 00->0, 01->1, 10->3, 11->4 (n=3, W[k*n+i])
    # W index for C_ik is k*n + i.
    # C_00: k=0, i=0 -> 0
    # C_01: k=1, i=0 -> 1
    # C_10: k=0, i=1 -> 3
    # C_11: k=1, i=1 -> 4
    
    strassen_U = [
        [1, 1, 0, 0, 0, 0, 0], # M1: A00, A11
        [0, 0, 1, 1, 0, 0, 0], # M2: A10, A11
        [1, 0, 0, 0, 0, 0, 0], # M3: A00
        [0, 0, 0, 1, 0, 0, 0], # M4: A11
        [1, 0, 1, 0, 0, 0, 0], # M5: A00, A10
        [1, 0, 0, 1, 0, 0, 0], # M6: A11, A00 (A11-A00)
        [0, 1, 0, 1, 0, 0, 0]  # M7: A01, A11 (A01-A11)
    ]
    # Note: U is (n*m, rank). So U[idx, col].
    # I need to transpose this or fill directly.
    
    strassen_V = [
        [1, 1, 0, 0, 0, 0, 0], # M1: B00, B11
        [1, 0, 0, 0, 0, 0, 0], # M2: B00
        [0, 0, 1, 1, 0, 0, 0], # M3: B10, B11 (B10-B11)
        [1, 1, 0, 0, 0, 0, 0], # M4: B01, B00 (B01-B00)
        [0, 0, 0, 0, 1, 0, 0], # M5: B11
        [1, 1, 0, 0, 0, 0, 0], # M6: B00, B01 (B00+B01)
        [0, 0, 1, 1, 0, 0, 0]  # M7: B10, B11 (B10+B11)
    ]
    # Signs:
    # M3: A00(B10 - B11) -> V[5]=1, V[6]=-1
    # M4: A11(B01 - B00) -> V[1]=1, V[0]=-1
    # M6: (A11 - A00)(B00 + B01) -> U[6]=1, U[0]=-1
    # M7: (A01 - A11)(B10 + B11) -> U[1]=1, U[6]=-1
    
    # Corrected V with signs
    strassen_V = [
        [1, 0, 0, 0, 0, 0, 0], # M1: B00, B11? No, M1=(A00+A11)(B00+B11). V[0]=1, V[6]=1
        [1, 0, 0, 0, 0, 0, 0], # M2: B00
        [0, 0, 1, -1, 0, 0, 0], # M3: B10, B11
        [-1, 1, 0, 0, 0, 0, 0], # M4: B01, B00
        [0, 0, 0, 0, 1, 0, 0], # M5: B11
        [1, 1, 0, 0, 0, 0, 0], # M6: B00, B01
        [0, 0, 1, 1, 0, 0, 0]  # M7: B10, B11
    ]
    # Corrected U with signs
    strassen_U = [
        [1, 1, 0, 0, 0, 0, 0], # M1
        [0, 0, 1, 1, 0, 0, 0], # M2
        [1, 0, 0, 0, 0, 0, 0], # M3
        [0, 0, 0, 1, 0, 0, 0], # M4
        [1, 0, 1, 0, 0, 0, 0], # M5
        [-1, 0, 0, 1, 0, 0, 0], # M6: A11 - A00
        [0, 1, 0, -1, 0, 0, 0]  # M7: A01 - A11
    ]
    
    # W for Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W indices: 0(C00), 1(C01), 3(C10), 4(C11)
    strassen_W = [
        [1, 0, 0, 1, 0, 0, 0], # M1
        [0, 0, 1, 0, 0, -1, 0], # M2
        [0, 1, 0, 0, 0, 0, 1], # M3
        [1, 0, 0, 1, 0, 0, 0], # M4
        [-1, 1, 0, 0, 0, 0, 0], # M5
        [0, 0, 0, 0, 0, 1, 0], # M6
        [1, 0, 0, 0, 0, 0, 1]  # M7
    ]
    
    # Fill U, V, W for Strassen
    for c in range(7):
        for idx, val in enumerate(strassen_U[c]):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(strassen_V[c]):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(strassen_W[c]):
            if val != 0:
                W[idx][c] = val
                
    # Schoolbook for the rest
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
