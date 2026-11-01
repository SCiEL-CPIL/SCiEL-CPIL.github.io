DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    rank = 15
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: Schoolbook for j=0
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        U[i * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][0] = 1
            
    # Strassen-Winograd parameters for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # The current program uses a specific set of 7 terms.
    # I will reuse the logic from the current program for the two 2x2x2 blocks.
    
    strassen_U = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
    ]
    strassen_V = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
    ]
    strassen_W = [
        {(0,0):1, (1,1):1},
        {(0,1):1},
        {(1,0):1},
        {(0,0):-1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):1},
        {(0,0):-1, (1,1):-1},
    ]
    
    # Block 1: j=1,2 (indices 1,2 in m=3)
    # This corresponds to A[i, 1..2] and B[1..2, k]
    # We map local indices (0,1) to global (1,2)
    for r in range(7):
        col = 1 + r
        for (ip, jp), val in strassen_U[r].items():
            # jp is local 0,1 -> global 1,2
            U[ip * m + (jp + 1)][col] = val
        for (jp, kp), val in strassen_V[r].items():
            # jp is local 0,1 -> global 1,2
            V[(jp + 1) * p + kp][col] = val
        for (kp, ip), val in strassen_W[r].items():
            W[kp * n + ip][col] = val
            
    # Block 2: j=1,2? No, the current program used j=1,2 for the first block and j=2,3?
    # m=3. Indices 0, 1, 2.
    # Block 1 used jp+1 -> 1, 2.
    # Block 2 used jp+1 -> 1, 2?
    # In the current program:
    # Loop 1: U[ip*3 + (jp+1)], V[(jp+1)*4 + kp], W[kp*2 + ip]
    # Loop 2: U[ip*3 + (jp+1)], V[(jp+1)*4 + (kp+2)], W[(kp+2)*2 + ip]
    # This covers k=0,1 in Block 1 and k=2,3 in Block 2.
    # So it splits the output dimension p=4 into two 2x2 blocks.
    
    for r in range(7):
        col = 8 + r
        for (ip, jp), val in strassen_U[r].items():
            U[ip * m + (jp + 1)][col] = val
        for (jp, kp), val in strassen_V[r].items():
            V[(jp + 1) * p + (kp + 2)][col] = val
        for (kp, ip), val in strassen_W[r].items():
            W[(kp + 2) * n + ip][col] = val

    return {"U": U, "V": V, "W": W}
