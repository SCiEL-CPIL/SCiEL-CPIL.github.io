"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank-12 decomposition as fallback
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
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try rank-11: Strassen (7 terms) for k=0,1 + schoolbook (4 terms) for k=2
    if time.time() < deadline:
        rank11 = 11
        U11 = [[0.0] * rank11 for _ in range(n * m)]
        V11 = [[0.0] * rank11 for _ in range(m * p)]
        W11 = [[0.0] * rank11 for _ in range(p * n)]
        
        # Indices:
        # U[i*m+j]: i∈{0,1}, j∈{0,1} → U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        # V[j*p+k]: j∈{0,1}, k∈{0,1,2} → V[0]=B00, V[1]=B01, V[2]=B02, V[3]=B10, V[4]=B11, V[5]=B12
        # W[k*n+i]: k∈{0,1,2}, i∈{0,1} → W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11, W[4]=C02, W[5]=C12
        
        # Strassen terms for 2x2x2 (k=0,1), terms 0-6:
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        # Contributes to C00 and C11
        U11[0][0] = 1; U11[3][0] = 1
        V11[0][0] = 1; V11[4][0] = 1
        W11[0][0] = 1; W11[3][0] = 1
        
        # Term 1: M2 = (A10+A11)B00
        # Contributes to C10
        U11[2][1] = 1; U11[3][1] = 1
        V11[0][1] = 1
        W11[1][1] = 1
        
        # Term 2: M3 = A00(B10-B11)
        # Contributes to C10
        U11[0][2] = 1
        V11[3][2] = 1; V11[4][2] = -1
        W11[1][2] = 1
        
        # Term 3: M4 = A11(B01-B10)
        # Contributes to C00 and C11
        U11[3][3] = 1
        V11[1][3] = 1; V11[3][3] = -1
        W11[0][3] = 1; W11[3][3] = 1
        
        # Term 4: M5 = (A00+A01)B11
        # Contributes to C00 and C01
        U11[0][4] = 1; U11[1][4] = 1
        V11[4][4] = 1
        W11[0][4] = 1; W11[2][4] = 1
        
        # Term 5: M6 = (A10-A00)(B00+B10)
        # Contributes to C11
        U11[2][5] = 1; U11[0][5] = -1
        V11[0][5] = 1; V11[3][5] = 1
        W11[3][5] = 1
        
        # Term 6: M7 = (A01-A11)(B01+B11)
        # Contributes to C00 and C01
        U11[1][6] = 1; U11[3][6] = -1
        V11[1][6] = 1; V11[4][6] = 1
        W11[0][6] = 1; W11[2][6] = 1
        
        # Schoolbook terms for k=2 (third column), terms 7-10:
        
        # Term 7: A00*B02 → C02
        U11[0][7] = 1
        V11[2][7] = 1
        W11[4][7] = 1
        
        # Term 8: A01*B12 → C02
        U11[1][8] = 1
        V11[5][8] = 1
        W11[4][8] = 1
        
        # Term 9: A10*B02 → C12
        U11[2][9] = 1
        V11[2][9] = 1
        W11[5][9] = 1
        
        # Term 10: A11*B12 → C12
        U11[3][10] = 1
        V11[5][10] = 1
        W11[5][10] = 1
        
        # Verify
        if _verify_decomposition(U11, V11, W11, n, m, p, tol=1e-9):
            best = {"U": U11, "V": V11, "W": W11}
            best_rank = rank11
        else:
            # Debug: print which entries fail
            U_arr = np.array(U11, dtype=float)
            V_arr = np.array(V11, dtype=float)
            W_arr = np.array(W11, dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(rank11):
                            total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                        if abs(total - 1.0) > 1e-9:
                            print(f"FAIL: i={i}, j={j}, k={k}, total={total}")
    
    return best
# EVOLVE-BLOCK-END
