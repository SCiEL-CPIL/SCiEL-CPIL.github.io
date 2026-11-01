import numpy as np

def construct():
    n, m, p = 2, 2, 4
    R = 14
    
    # Strassen for 2x2 * 2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 * 2x4, split B into B_left (cols 0,1) and B_right (cols 2,3)
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    # Total output C is 2x4
    
    # We use 7 products for left, 7 for right = 14 total
    # But we can potentially share some. For now, use 14.
    
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # i in 0,1; j in 0,1; k in 0,1,2,3
    
    # Left block (k=0,1): B_left[0,0]=B[0,0], B_left[0,1]=B[0,1], B_left[1,0]=B[1,0], B_left[1,1]=B[1,1]
    # Right block (k=2,3): B_right[0,0]=B[0,2], B_right[0,1]=B[0,3], B_right[1,0]=B[1,2], B_right[1,1]=B[1,3]
    
    # Strassen products for left block:
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    # M3: A[0,0] * (B[1,0]-B[1,1])
    # M4: A[1,1] * (B[0,1]-B[1,0])
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
    # M7: (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
    
    # M1
    r = 0
    U[0*2+0, r] = 1  # A[0,0]
    U[1*2+1, r] = 1  # A[1,1]
    V[0*4+0, r] = 1  # B[0,0]
    V[1*4+1, r] = 1  # B[1,1]
    # W: C[0,0] += M1, C[0,1] += M1? No.
    # C_left[0,0] = M1+M4-M5+M7
    # C_left[0,1] = M3+M5
    # C_left[1,0] = M2+M4
    # C_left[1,1] = M1-M2+M3+M6
    # Output index: k=0 -> col 0, k=1 -> col 1
    # W[k*n+i, r]: k=0,i=0 -> W[0]; k=0,i=1 -> W[1]; k=1,i=0 -> W[2]; k=1,i=1 -> W[3]
    W[0*2+0, r] = 1   # C[0,0]
    W[1*2+1, r] = 1   # C[1,1]
    
    # M2
    r = 1
    U[1*2+0, r] = 1  # A[1,0]
    U[1*2+1, r] = 1  # A[1,1]
    V[0*4+0, r] = 1  # B[0,0]
    W[1*2+0, r] = 1   # C[1,0]
    W[1*2+1, r] = -1  # C[1,1]
    
    # M3
    r = 2
    U[0*2+0, r] = 1  # A[0,0]
    V[1*4+0, r] = 1  # B[1,0]
    V[1*4+1, r] = -1 # B[1,1]
    W[0*2+1, r] = 1   # C[0,1]
    W[1*2+1, r] = 1   # C[1,1]
    
    # M4
    r = 3
    U[1*2+1, r] = 1  # A[1,1]
    V[0*4+1, r] = 1  # B[0,1]
    V[1*4+0, r] = -1 # B[1,0]
    W[0*2+0, r] = 1   # C[0,0]
    W[1*2+0, r] = 1   # C[1,0]
    
    # M5
    r = 4
    U[0*2+0, r] = 1  # A[0,0]
    U[0*2+1, r] = 1  # A[0,1]
    V[1*4+1, r] = 1  # B[1,1]
    W[0*2+0, r] = -1  # C[0,0]
    W[0*2+1, r] = 1   # C[0,1]
    
    # M6
    r = 5
    U[1*2+0, r] = 1  # A[1,0]
    U[0*2+0, r] = -1 # A[0,0]
    V[0*4+0, r] = 1  # B[0,0]
    V[1*4+0, r] = 1  # B[1,0]
    W[1*2+1, r] = 1   # C[1,1]
    
    # M7
    r = 6
    U[0*2+1, r] = 1  # A[0,1]
    U[1*2+1, r] = -1 # A[1,1]
    V[0*4+1, r] = 1  # B[0,1]
    V[1*4+1, r] = 1  # B[1,1]
    W[0*2+0, r] = 1   # C[0,0]
    
    # Right block (k=2,3): B_right[0,0]=B[0,2], B_right[0,1]=B[0,3], B_right[1,0]=B[1,2], B_right[1,1]=B[1,3]
    # Same Strassen structure but with B indices shifted by 2
    
    # M8 (right M1)
    r = 7
    U[0*2+0, r] = 1  # A[0,0]
    U[1*2+1, r] = 1  # A[1,1]
    V[0*4+2, r] = 1  # B[0,2]
    V[1*4+3, r] = 1  # B[1,3]
    W[2*2+0, r] = 1   # C[0,2]
    W[3*2+1, r] = 1   # C[1,3]
    
    # M9 (right M2)
    r = 8
    U[1*2+0, r] = 1  # A[1,0]
    U[1*2+1, r] = 1  # A[1,1]
    V[0*4+2, r] = 1  # B[0,2]
    W[2*2+0, r] = 1   # C[0,2]
    W[3*2+1, r] = -1  # C[1,3]
    
    # M10 (right M3)
    r = 9
    U[0*2+0, r] = 1  # A[0,0]
    V[1*4+2, r] = 1  # B[1,2]
    V[1*4+3, r] = -1 # B[1,3]
    W[2*2+1, r] = 1   # C[0,3]
    W[3*2+1, r] = 1   # C[1,3]
    
    # M11 (right M4)
    r = 10
    U[1*2+1, r] = 1  # A[1,1]
    V[0*4+3, r] = 1  # B[0,3]
    V[1*4+2, r] = -1 # B[1,2]
    W[2*2+0, r] = 1   # C[0,2]
    W[3*2+0, r] = 1   # C[1,2]
    
    # M12 (right M5)
    r = 11
    U[0*2+0, r] = 1  # A[0,0]
    U[0*2+1, r] = 1  # A[0,1]
    V[1*4+3, r] = 1  # B[1,3]
    W[2*2+0, r] = -1  # C[0,2]
    W[2*2+1, r] = 1   # C[0,3]
    
    # M13 (right M6)
    r = 12
    U[1*2+0, r] = 1  # A[1,0]
    U[0*2+0, r] = -1 # A[0,0]
    V[0*4+2, r] = 1  # B[0,2]
    V[1*4+2, r] = 1  # B[1,2]
    W[3*2+1, r] = 1   # C[1,3]
    
    # M14 (right M7)
    r = 13
    U[0*2+1, r] = 1  # A[0,1]
    U[1*2+1, r] = -1 # A[1,1]
    V[0*4+3, r] = 1  # B[0,3]
    V[1*4+3, r] = 1  # B[1,3]
    W[2*2+0, r] = 1   # C[0,2]
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to trivial rank 16
        R_trivial = n * m * p
        U_t = np.zeros((n*m, R_trivial))
        V_t = np.zeros((m*p, R_trivial))
        W_t = np.zeros((p*n, R_trivial))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i*m+j, col] = 1
                    V_t[j*p+k, col] = 1
                    W_t[k*n+i, col] = 1
                    col += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
