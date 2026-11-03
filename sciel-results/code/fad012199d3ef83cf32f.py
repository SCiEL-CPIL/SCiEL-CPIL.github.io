import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 50
    # Part 1: j=0 (2x1x9) -> Rank 18
    # Part 2: j in {1,2} (2x2x9) -> Split k into 4 blocks of 2, 1 block of 1
    #   4 * (2x2x2 Strassen rank 7) + 1 * (2x2x1 schoolbook rank 4) = 28 + 4 = 32
    # Total Rank = 18 + 32 = 50
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    # i in 0..1, k in 0..8
    for i in range(n):
        for k in range(p):
            # U[i, 0] = 1
            U[i * m + 0][col] = 1.0
            # V[0, k] = 1
            V[0 * p + k][col] = 1.0
            # W[k, i] = 1 (Note: W index is k*n + i)
            W[k * n + i][col] = 1.0
            col += 1
            
    # Part 2: j in {1, 2}
    # Map local j' in {0,1} to global j = j' + 1
    # Split k into blocks
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)]
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U coeffs (local i, j')
    # V coeffs (local j', k')
    # W coeffs (local i, k')
    
    strassen_U = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    strassen_V = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    strassen_W = [
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }, # M1 -> C00
        { (0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0 }, # M2 -> C10
        { (0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0 }, # M3 -> C01
        { (0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0 }, # M4 -> C00, C10
        { (0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0 },# M5 -> C00, C01
        { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1 }, # M6 -> C11
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1 }  # M7 -> C00, C11
    ]
    # Wait, C11 = M1 - M2 + M3 + M6
    # My W above:
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=1, C10=1
    # M5: C00=-1, C01=1
    # M6: C11=1
    # M7: C00=1, C11=1
    
    # Let's re-verify C11
    # C11 = M1 - M2 + M3 + M6
    # M1: +1
    # M2: -1
    # M3: +1
    # M6: +1
    # Sum: 1 - 1 + 1 + 1 = 2? No.
    # Strassen:
    # C11 = M1 - M2 + M3 + M6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M6 = (A10-A00)(B00+B01)
    # Let's check C11 = A10 B01 + A11 B11
    # This formula is standard.
    
    # Correct W assignments:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_W_correct = [
        { (0,0): 1, (1,1): 1 }, # M1: C00, C11
        { (1,0): 1, (1,1): -1 },# M2: C10, -C11
        { (0,1): 1, (1,1): 1 }, # M3: C01, C11
        { (0,0): 1, (1,0): 1 }, # M4: C00, C10
        { (0,0): -1, (0,1): 1 },# M5: -C00, C01
        { (1,1): 1 },           # M6: C11
        { (0,0): 1, (1,1): 1 }  # M7: C00, C11
    ]
    # Check C11: 1 - 1 + 1 + 1 = 2?
    # Ah, M6 is (A10-A00)(B00+B01).
    # In C11, the coefficient of M6 is +1.
    # Let's re-read Strassen.
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # My indices:
    # i=0,1 (rows), j=0,1 (cols), k=0,1 (cols)
    # A is i,j. B is j,k. C is i,k.
    # c11 (i=0,k=0) = M1 + M4 - M5 + M7
    # c12 (i=0,k=1) = M3 + M5
    # c21 (i=1,k=0) = M2 + M4
    # c22 (i=1,k=1) = M1 - M2 + M3 + M6
    
    # My W above:
    # M1: (0,0)=1, (1,1)=1. Correct.
    # M2: (1,0)=1, (1,1)=-1. Correct.
    # M3: (0,1)=1, (1,1)=1. Correct.
    # M4: (0,0)=1, (1,0)=1. Correct.
    # M5: (0,0)=-1, (0,1)=1. Correct.
    # M6: (1,1)=1. Correct.
    # M7: (0,0)=1, (1,1)=1. Correct.
    
    # Process k blocks
    for k_start, k_end in k_blocks:
        k_size = k_end - k_start
        if k_size == 2:
            # Strassen
            for s in range(7):
                # U
                for (li, lj), val in strassen_U[s].items():
                    global_i = li
                    global_j = lj + 1
                    U[global_i * m + global_j][col] += val
                # V
                for (lj, lk), val in strassen_V[s].items():
                    global_j = lj + 1
                    global_k = k_start + lk
                    V[global_j * p + global_k][col] += val
                # W
                for (li, lk), val in strassen_W_correct[s].items():
                    global_i = li
                    global_k = k_start + lk
                    W[global_k * n + global_i][col] += val
                col += 1
        elif k_size == 1:
            # Schoolbook 2x2x1
            # j in {1,2}, k = k_start
            for i in range(2):
                for j_local in range(2):
                    global_j = j_local + 1
                    global_k = k_start
                    U[i * m + global_j][col] = 1.0
                    V[global_j * p + global_k][col] = 1.0
                    W[global_k * n + i][col] = 1.0
                    col += 1

    # Verification
    # Check sum_r W[r][i][k] U[r][i'][j'] V[r][j'][k'] == delta(i',i) delta(k',k)
    # Note: W index is k*n + i. U is i*m + j. V is j*p + k.
    
    for i in range(n):
        for k in range(p):
            for ip in range(n):
                for jp in range(m):
                    for kp in range(p):
                        expected = 1.0 if (ip == i and kp == k) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += W[k * n + i][r] * U[ip * m + jp][r] * V[jp * p + kp][r]
                        if abs(actual - expected) > 1e-9:
                            return {"U": U, "V": W, "W": W} # Error case, return invalid or raise
                            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
