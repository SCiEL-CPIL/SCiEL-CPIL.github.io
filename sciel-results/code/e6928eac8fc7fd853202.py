import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(j_start, k_start):
        nonlocal col
        # A: i in {0,1}, j in {j_start, j_start+1}
        # B: j in {j_start, j_start+1}, k in {k_start, k_start+1}
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i*m + j_start][col] += 1
        U[i*m + j_start+1][col] += 1 # Wait, A11 is i=1, j=j_start+1
        # Correct indices:
        # A00: i=0, j=j_start
        # A11: i=1, j=j_start+1
        # A10: i=1, j=j_start
        # A01: i=0, j=j_start+1
        
        # Let's rewrite carefully.
        
        # M1
        U[0*m + j_start][col] += 1
        U[1*m + j_start+1][col] += 1
        V[j_start*p + k_start][col] += 1
        V[j_start+1*p + k_start+1][col] += 1
        W[k_start*n + 0][col] += 1
        W[k_start+1*n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[1*m + j_start][col] += 1
        U[1*m + j_start+1][col] += 1
        V[j_start*p + k_start][col] += 1
        W[k_start*n + 1][col] += 1
        W[k_start+1*n + 1][col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[0*m + j_start][col] += 1
        V[j_start*p + k_start+1][col] += 1
        V[j_start+1*p + k_start+1][col] -= 1
        W[k_start+1*n + 0][col] += 1
        W[k_start+1*n + 1][col] += 1
        col += 1
        
        # M4 = A11(B10 + B11)
        U[1*m + j_start+1][col] += 1
        V[j_start+1*p + k_start][col] += 1
        V[j_start+1*p + k_start+1][col] += 1
        W[k_start*n + 0][col] += 1
        W[k_start+1*n + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0*m + j_start][col] += 1
        U[0*m + j_start+1][col] += 1
        V[j_start+1*p + k_start+1][col] += 1
        W[k_start*n + 0][col] -= 1
        W[k_start+1*n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[1*m + j_start][col] += 1
        U[0*m + j_start][col] -= 1
        V[j_start*p + k_start][col] += 1
        V[j_start*p + k_start+1][col] += 1
        W[k_start+1*n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[0*m + j_start+1][col] += 1
        U[1*m + j_start+1][col] -= 1
        V[j_start+1*p + k_start][col] += 1
        V[j_start+1*p + k_start+1][col] += 1
        W[k_start*n + 0][col] -= 1
        W[k_start*n + 1][col] += 1
        col += 1

    def add_schoolbook(j_start, k):
        nonlocal col
        # A: i in {0,1}, j in {j_start, j_start+1}
        # B: j in {j_start, j_start+1}, k
        # C: k, i
        
        # T1 = A00 B0k
        U[0*m + j_start][col] += 1
        V[j_start*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # T2 = A10 B1k
        U[1*m + j_start][col] += 1
        V[j_start+1*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # T3 = A01 B0k
        U[0*m + j_start+1][col] += 1
        V[j_start*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
        # T4 = A11 B1k
        U[1*m + j_start+1][col] += 1
        V[j_start+1*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1

    # Group 1: A_L (j=0,1), B_T (j=0,1)
    add_strassen(0, 0) # k=0,1
    add_strassen(0, 2) # k=2,3
    add_schoolbook(0, 4) # k=4
    
    # Group 2: A_R (j=2,3), B_B (j=2,3)
    add_strassen(2, 0) # k=0,1
    add_strassen(2, 2) # k=2,3
    add_schoolbook(2, 4) # k=4

    return {"U": U, "V": V, "W": W}
