import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # 1. Build Strassen 2x2x2 decomposition (Rank 7)
    # U2: (4, 7) -> A (2x2)
    # V2: (4, 7) -> B (2x2)
    # W2: (4, 7) -> C (2x2)
    # Indices:
    # U2[i*2+k], V2[k*2+j], W2[j*2+i]
    
    U2 = np.zeros((4, 7), dtype=np.int8)
    V2 = np.zeros((4, 7), dtype=np.int8)
    W2 = np.zeros((4, 7), dtype=np.int8)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # A00 idx 0, A11 idx 3
    # B00 idx 0, B11 idx 3
    # C00 idx 0, C11 idx 3
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    # A10 idx 2, A11 idx 3
    # B00 idx 0
    # C10 idx 1 (j=1,i=0 -> 1*2+0=2? No. C is 2x2. C[j,i]. 
    # Wait, W index is j*n+i. For 2x2, n=2.
    # C00: j=0,i=0 -> 0.
    # C01: j=0,i=1 -> 1.
    # C10: j=1,i=0 -> 2.
    # C11: j=1,i=1 -> 3.
    #
    # Let's re-verify Strassen formulas with C[j,i] indexing.
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In our indexing W[j*n+i]:
    # C00 is W[0]
    # C01 is W[1]
    # C10 is W[2]
    # C11 is W[3]
    
    # r=1: M2 = (A10+A11)B00
    # Contributes to C10 (W[2]) with +1 and C11 (W[3]) with -1?
    # Let's check standard Strassen again.
    # M2 = (A10+A11)B00
    # C10 = M2 + M4 ... so C10 gets +M2.
    # C11 = M1 - M2 + ... so C11 gets -M2.
    # So W[2] += 1, W[3] -= 1.
    
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    # A00 idx 0
    # B01 idx 1 (k=0, j=1 -> 0*2+1=1), B11 idx 3
    # Contributes to C01 (W[1]) with +1 and C11 (W[3]) with +1?
    # C01 = M3 + M5 ... so C01 gets +M3.
    # C11 = M1 - M2 + M3 + M6 ... so C11 gets +M3.
    # So W[1] += 1, W[3] += 1.
    
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    # A11 idx 3
    # B10 idx 2 (k=1, j=0 -> 1*2+0=2), B00 idx 0
    # Contributes to C00 (W[0]) with +1 and C10 (W[2]) with +1?
    # C00 = M1 + M4 ... so C00 gets +M4.
    # C10 = M2 + M4 ... so C10 gets +M4.
    # So W[0] += 1, W[2] += 1.
    
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # r=4: M5 = (A00+A01)B11
    # A00 idx 0, A01 idx 1
    # B11 idx 3
    # Contributes to C00 (W[0]) with -1 and C01 (W[1]) with +1?
    # C00 = M1 + M4 - M5 ... so C00 gets -M5.
    # C01 = M3 + M5 ... so C01 gets +M5.
    # So W[0] -= 1, W[1] += 1.
    
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    # A10 idx 2, A00 idx 0
    # B00 idx 0, B01 idx 1
    # Contributes to C11 (W[3]) with +1?
    # C11 = M1 - M2 + M3 + M6 ... so C11 gets +M6.
    # So W[3] += 1.
    
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    # A01 idx 1, A11 idx 3
    # B10 idx 2, B11 idx 3
    # Contributes to C00 (W[0]) with +1?
    # C00 = M1 + M4 - M5 + M7 ... so C00 gets +M7.
    # So W[0] += 1.
    
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 2. Build 4x4x4 decomposition (Rank 49) via Tensor Product
    U4 = np.zeros((16, 49), dtype=np.int16)
    V4 = np.zeros((16, 49), dtype=np.int16)
    W4 = np.zeros((16, 49), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for k in range(4):
                    i0, i1 = i // 2, i % 2
                    k0, k1 = k // 2, k % 2
                    # U2 index for A[i,k] block c0: (i//2)*2 + (k//2)
                    # U2 index for A[i,k] block c1: (i%2)*2 + (k%2)
                    U4[i*4+k, c] = U2[i0*2+k0, c0] * U2[i1*2+k1, c1]
            
            for k in range(4):
                for j in range(4):
                    k0, k1 = k // 2, k % 2
                    j0, j1 = j // 2, j % 2
                    V4[k*4+j, c] = V2[k0*2+j0, c0] * V2[k1*2+j1, c1]
            
            for j in range(4):
                for i in range(4):
                    j0, j1 = j // 2, j % 2
                    i0, i1 = i // 2, i % 2
                    W4[j*4+i, c] = W2[j0*2+i0, c0] * W2[j1*2+i1, c1]

    # 3. Build 4x4x8 decomposition (Rank 98)
    # Split into two 4x4x4 blocks
    rank = 98
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)
    
    for b in range(2):
        col_start = b * 49
        col_end = (b + 1) * 49
        
        # U is identical for both blocks
        U[:, col_start:col_end] = U4
        
        # V: shape (32, 98). Index k*8+j.
        # For block b, j ranges from b*4 to b*4+3.
        # V[k*8+j, col] = V4[k*4 + (j - b*4), col - col_start]
        for k in range(4):
            for j in range(4):
                j_actual = b * 4 + j
                V[k*8 + j_actual, col_start:col_end] = V4[k*4 + j, :]
        
        # W: shape (32, 98). Index j*4+i.
        # For block b, j ranges from b*4 to b*4+3.
        # W[j*4+i, col] = W4[(j - b*4)*4 + i, col - col_start]
        for j in range(4):
            j_actual = b * 4 + j
            for i in range(4):
                W[j_actual*4 + i, col_start:col_end] = W4[j*4 + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 4, 4, 8
    # Check a few random entries
    import random
    random.seed(42)
    errors = 0
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        # Target: 1 if this is a valid term in the sum? 
        # No, the tensor T[i*m+k, j*p+k? No.
        # Tensor T[A_idx, B_idx, C_idx] = 1 if A_idx=i*m+k, B_idx=k*p+j, C_idx=j*n+i.
        # We are checking the sum over r of U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
        # This sum should be 1 for all i,j,k.
        
        a_idx = i * m + k
        b_idx = k * p + j
        c_idx = j * n + i
        
        val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
        if abs(val - 1.0) > 1e-9:
            errors += 1
            print(f"Error at i={i}, j={j}, k={k}: val={val}")
    
    if errors == 0:
        print("Verification passed.")
    else:
        print(f"Verification failed with {errors} errors.")
