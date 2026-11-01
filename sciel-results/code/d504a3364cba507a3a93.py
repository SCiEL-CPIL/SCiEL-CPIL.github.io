import numpy as np

def construct():
    n, m, p = 2, 2, 5
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: Strassen for B cols 0, 1
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->5, 11->6
    # C indices: 00->0, 01->2, 10->1, 11->3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, col] = 1; U[3, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[2, col] = 1; U[3, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[3, col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0, col] = 1
    V[1, col] = 1; V[6, col] = -1
    W[2, col] = 1; W[3, col] = 1
    col += 1
    
    # M4 = A11(B10-B01)
    U[3, col] = 1
    V[5, col] = 1; V[1, col] = -1
    W[0, col] = -1; W[2, col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0, col] = 1; U[1, col] = 1
    V[6, col] = 1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[1, col] = 1; W[2, col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, col] = 1; U[3, col] = -1
    V[5, col] = 1; V[6, col] = 1
    W[0, col] = -1; W[3, col] = 1
    col += 1
    
    # Part 2: Strassen for B cols 2, 3
    # B indices: 02->2, 03->3, 12->7, 13->8
    # C indices: 02->4, 03->6, 12->5, 13->7
    
    # M8 = (A00+A11)(B02+B13)
    U[0, col] = 1; U[3, col] = 1
    V[2, col] = 1; V[8, col] = 1
    W[4, col] = 1; W[7, col] = 1
    col += 1
    
    # M9 = (A10+A11)B02
    U[2, col] = 1; U[3, col] = 1
    V[2, col] = 1
    W[5, col] = 1; W[7, col] = -1
    col += 1
    
    # M10 = A00(B03-B13)
    U[0, col] = 1
    V[3, col] = 1; V[8, col] = -1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    # M11 = A11(B12-B03)
    U[3, col] = 1
    V[7, col] = 1; V[3, col] = -1
    W[4, col] = -1; W[6, col] = 1
    col += 1
    
    # M12 = (A00+A01)B13
    U[0, col] = 1; U[1, col] = 1
    V[8, col] = 1
    W[4, col] = 1; W[5, col] = 1
    col += 1
    
    # M13 = (A10-A00)(B02+B03)
    U[2, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[5, col] = 1; W[6, col] = 1
    col += 1
    
    # M14 = (A01-A11)(B12+B13)
    U[1, col] = 1; U[3, col] = -1
    V[7, col] = 1; V[8, col] = 1
    W[4, col] = -1; W[7, col] = 1
    col += 1
    
    # Part 3: Schoolbook for B col 4
    # B indices: 04->4, 14->9
    # C indices: 04->8, 14->9
    
    # M15 = A00 B04
    U[0, col] = 1
    V[4, col] = 1
    W[8, col] = 1
    col += 1
    
    # M16 = A01 B14
    U[1, col] = 1
    V[9, col] = 1
    W[8, col] = 1
    col += 1
    
    # M17 = A10 B04
    U[2, col] = 1
    V[4, col] = 1
    W[9, col] = 1
    col += 1
    
    # M18 = A11 B14
    U[3, col] = 1
    V[9, col] = 1
    W[9, col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
