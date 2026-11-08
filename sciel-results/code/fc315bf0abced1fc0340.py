import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: i*m+j (i in 0..n-1, j in 0..m-1)
    # B indices: j*p+k (j in 0..m-1, k in 0..p-1)
    # C indices: k*n+i (k in 0..p-1, i in 0..n-1)
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 3, 4
    # B_T uses B[0,0..8], B[1,0..8] -> V indices 0..8, 9..17
    
    col = 0
    # 4 blocks of 2x2 (k=0,1; 2,3; 4,5; 6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0   # B[0, k0]
        v_f = 0 * p + k1   # B[0, k1]
        v_g = 1 * p + k0   # B[1, k0]
        v_h = 1 * p + k1   # B[1, k1]
        
        u_a = 0 * m + 0   # A[0, 0]
        u_b = 0 * m + 1   # A[0, 1]
        u_c = 1 * m + 0   # A[1, 0]
        u_d = 1 * m + 1   # A[1, 1]
        
        # Strassen 7-product decomposition for 2x2 * 2x2
        # M1 = (A00 + A11)(B00 + B11)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = A11(B01 - B00)
        U[u_d][col+1] += 1
        V[v_f][col+1] += 1
        V[v_e][col+1] -= 1
        
        # M3 = (A00 - A10)B10
        U[u_a][col+2] += 1
        U[u_c][col+2] -= 1
        V[v_g][col+2] += 1
        
        # M4 = A00(B11 - B01)
        U[u_a][col+3] += 1
        V[v_h][col+3] += 1
        V[v_f][col+3] -= 1
        
        # M5 = (A10 + A00)(B00 + B01)
        U[u_a][col+4] += 1
        U[u_c][col+4] += 1
        V[v_e][col+4] += 1
        V[v_f][col+4] += 1
        
        # M6 = (A11 - A01)(B10 + B11)
        U[u_d][col+5] += 1
        U[u_b][col+5] -= 1
        V[v_g][col+5] += 1
        V[v_h][col+5] += 1
        
        # M7 = (A01 - A11)(B01 + B10)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_f][col+6] += 1
        V[v_g][col+6] += 1
        
        # W contributions: C[i,k] = sum_j A[i,j]*B[j,k]
        # C[0,k0] = A[0,0]*B[0,k0] + A[0,1]*B[1,k0]
        # C[1,k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0]
        # C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1]
        # C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1]
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0]: M1 + M3 + M5 - M6 - M7
        # Actually let me re-derive:
        # C[0,k0] = A00*B00 + A01*B10
        # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # M2 = A11(B01-B00) = A11*B01 - A11*B00
        # M3 = (A00-A10)B10 = A00*B10 - A10*B10
        # M4 = A00(B11-B01) = A00*B11 - A00*B01
        # M5 = (A10+A00)(B00+B01) = A10*B00 + A10*B01 + A00*B00 + A00*B01
        # M6 = (A11-A01)(B10+B11) = A11*B10 + A11*B11 - A01*B10 - A01*B11
        # M7 = (A01-A11)(B01+B10) = A01*B01 + A01*B10 - A11*B01 - A11*B10
        
        # C[0,k0] = A00*B00 + A01*B10
        # = M1 - M2 - M3 - M4 - M5 + M6 + M7 ... let me just verify with the standard formula
        
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3
        # C10 = M5 + M6
        # C11 = M1 - M2 + M3 + M4
        
        W[w_00][col] += 1     # M1
        W[w_00][col+3] += 1   # M4
        W[w_00][col+4] -= 1   # M5
        W[w_00][col+6] += 1   # M7
        
        W[w_01][col+1] += 1   # M2
        W[w_01][col+2] += 1   # M3
        
        W[w_10][col+4] += 1   # M5
        W[w_10][col+5] += 1   # M6
        
        W[w_11][col] += 1     # M1
        W[w_11][col+1] -= 1   # M2
        W[w_11][col+2] += 1   # M3
        W[w_11][col+3] += 1   # M4
        
        col += 7
    
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8   # B[0, 8]
    v_g = 1 * p + k8   # B[1, 8]
    
    u_a = 0 * m + 0   # A[0, 0]
    u_b = 0 * m + 1   # A[0, 1]
    u_c = 1 * m + 0   # A[1, 0]
    u_d = 1 * m + 1   # A[1, 1]
    
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    # Direct: 4 products
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    W[k8 * n + 0][col] += 1
    
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    W[k8 * n + 0][col+1] += 1
    
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    W[k8 * n + 1][col+2] += 1
    
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    W[k8 * n + 1][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A[0,2], A[1,2] -> U indices 2, 5
    # B_B uses B[2,0..8] -> V indices 18..26
    
    u_ar0 = 0 * m + 2   # A[0, 2]
    u_ar1 = 1 * m + 2   # A[1, 2]
    
    for k in range(9):
        v_b = 2 * p + k   # B[2, k]
        
        # C[0,k] += A[0,2]*B[2,k]
        # C[1,k] += A[1,2]*B[2,k]
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
