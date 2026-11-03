import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # 1. Construct 4x4x4 decomposition using Kronecker product of Strassen
    # Strassen 2x2x2 decomposition
    # Indices for 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U, V, W shapes (4, 7)
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to represent this as sum of rank-1 tensors.
    # Each M is a product of a linear combo of A and a linear combo of B.
    # So each M corresponds to 1 term in the decomposition.
    
    # Let's build the 2x2x2 U, V, W matrices.
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: M1
    # A part: A00 + A11 -> U[0,0]=1, U[3,0]=1
    # B part: B00 + B11 -> V[0,0]=1, V[3,0]=1
    # C part: C00 += 1, C11 += 1 -> W[0,0]=1, W[3,0]=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: M2
    # A part: A10 + A11 -> U[2,1]=1, U[3,1]=1
    # B part: B00 -> V[0,1]=1
    # C part: C10 += 1, C11 -= 1 -> W[2,1]=1, W[3,1]=-1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: M3
    # A part: A00 -> U[0,2]=1
    # B part: B01 - B11 -> V[1,2]=1, V[3,2]=-1
    # C part: C01 += 1, C11 += 1 -> W[1,2]=1, W[3,2]=1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: M4
    # A part: A11 -> U[3,3]=1
    # B part: B10 - B00 -> V[2,3]=1, V[0,3]=-1
    # C part: C00 += 1, C10 += 1 -> W[0,3]=1, W[2,3]=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: M5
    # A part: A00 + A10 -> U[0,4]=1, U[2,4]=1
    # B part: B11 -> V[3,4]=1
    # C part: C00 -= 1, C01 += 1 -> W[0,4]=-1, W[1,4]=1
    U2[0, 4] = 1; U2[2, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: M6
    # A part: A01 - A00 -> U[1,5]=1, U[0,5]=-1
    # B part: B00 + B10 -> V[0,5]=1, V[2,5]=1
    # C part: C11 += 1 -> W[3,5]=1
    U2[1, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: M7
    # A part: A10 - A11 -> U[2,6]=1, U[3,6]=-1
    # B part: B11 + B01 -> V[3,6]=1, V[1,6]=1
    # C part: C00 += 1 -> W[0,6]=1
    U2[2, 6] = 1; U2[3, 6] = -1
    V2[3, 6] = 1; V2[1, 6] = 1
    W2[0, 6] = 1
    
    # Now build 4x4x4 using Kronecker product of U2, V2, W2
    # U4 shape (16, 49)
    # V4 shape (16, 49)
    # W4 shape (16, 49)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U4 indices: (i1*2+i2)*2 + (j1*2+j2)
            # U2 indices: i1*2+j1, i2*2+j2
            # Actually, U2 is for 2x2.
            # The 4x4 matrix is a 2x2 block of 2x2 matrices.
            # A[i,j] where i = i1*2 + i2, j = j1*2 + j2.
            # U4[i,j, r] = U2[i1,j1, r1] * U2[i2,j2, r2]
            
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U2[i1*2+j1, r1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val2 = U2[i2*2+j2, r2]
                            if val2 == 0: continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U4[i*4+j, r] = val1 * val2
                            
            # Similarly for V4
            # V2 indices: j*2+k
            # V4 indices: (j1*2+j2)*2 + (k1*2+k2) ?
            # Wait, V is for B. B is 4x4.
            # B[j,k]. j = j1*2+j2, k = k1*2+k2.
            # V4[j,k, r] = V2[j1,k1, r1] * V2[j2,k2, r2]
            
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V2[j1*2+k1, r1]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            val2 = V2[j2*2+k2, r2]
                            if val2 == 0: continue
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V4[j*4+k, r] = val1 * val2
                            
            # Similarly for W4
            # W2 indices: k*2+i
            # W4 indices: (k1*2+k2)*2 + (i1*2+i2)
            # W4[k,i, r] = W2[k1,i1, r1] * W2[k2,i2, r2]
            
            for k1 in range(2):
                for i1 in range(2):
                    val1 = W2[k1*2+i1, r1]
                    if val1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            val2 = W2[k2*2+i2, r2]
                            if val2 == 0: continue
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W4[k*4+i, r] = val1 * val2

    # Now we have U4, V4, W4 for the first 4 columns of B (k=0..3)
    # We need to add the 5th column (k=4)
    # The 5th column involves B[j,4] and C[i,4]
    # C[i,4] = sum_j A[i,j] B[j,4]
    # This is 16 terms.
    
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank)) # m*p = 4*5 = 20
    W = np.zeros((20, rank)) # p*n = 5*4 = 20
    
    # Copy the 4x4x4 part
    # U4 indices: i*4+j (0..15)
    # V4 indices: j*4+k (0..15) -> but V is 4x5, so indices j*5+k
    # W4 indices: k*4+i (0..15) -> but W is 5x4, so indices k*4+i
    
    for r in range(49):
        # U
        U[:, r] = U4[:, r]
        
        # V
        # V4[j,k] is at index j*4+k
        # V[j,k] is at index j*5+k
        # k is 0..3
        for j in range(4):
            for k in range(4):
                V[j*5+k, r] = V4[j*4+k, r]
                
        # W
        # W4[k,i] is at index k*4+i
        # W[k,i] is at index k*4+i
        W[:, r] = W4[:, r]
        
    # Add the 5th column terms
    # For each i in 0..3, j in 0..3
    # Term: A[i,j] * B[j,4] -> C[i,4]
    # U[i*4+j, r] = 1
    # V[j*5+4, r] = 1
    # W[4*4+i, r] = 1
    
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1
            W[4*4+i, col] = 1
            col += 1
            
    # Convert to lists of lists as required
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
