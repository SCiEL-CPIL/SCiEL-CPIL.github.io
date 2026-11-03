"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 86 using:
    # 1. 4x4 Strassen (rank 49) for the top-left 4x4 block
    # 2. Schoolbook for the remaining terms involving the 5th index
    
    # --- Step 1: 4x4 Strassen Decomposition ---
    # Strassen for 2x2 has rank 7.
    # Strassen for 4x4 is built from 7 2x2 Strassen blocks? 
    # Actually, 4x4 Strassen rank is 49.
    # We can generate the 4x4 factors by applying the 2x2 Strassen logic to the 4 blocks of the 4x4 matrix.
    
    # 2x2 Strassen factors (u, v, w) for indices (i,j) in A, (j,k) in B, (k,i) in C
    # i,j,k in {0,1}
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    # We will build the 4x4 decomposition by iterating over the 4 2x2 blocks of the 4x4 matrix.
    # Let the 4x4 matrix A be composed of blocks A00, A01, A10, A11 (each 2x2).
    # The 4x4 Strassen decomposition is the tensor product of the 2x2 Strassen decomposition with itself?
    # No, it's recursive.
    # U_4x4[i, j, r] where i,j in 0..3.
    # r corresponds to a pair of 2x2 Strassen indices (r1, r2).
    
    # 2x2 Strassen U, V, W arrays
    # U2[i, j, r], V2[j, k, r], W2[k, i, r] for i,j,k in 0,1 and r in 0..6
    
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0: M1
    U2[0,0,0] = 1; U2[1,1,0] = 1
    V2[0,0,0] = 1; V2[1,1,0] = 1
    W2[0,0,0] = 1; W2[1,1,0] = 1
    
    # r=1: M2
    U2[1,0,1] = 1; U2[1,1,1] = 1
    V2[0,0,1] = 1
    W2[1,0,1] = 1; W2[1,1,1] = 1
    
    # r=2: M3
    U2[0,0,2] = 1
    V2[0,1,2] = 1; V2[1,1,2] = -1
    W2[0,1,2] = 1; W2[1,1,2] = 1
    
    # r=3: M4
    U2[1,1,3] = 1
    V2[1,0,3] = 1; V2[0,0,3] = -1
    W2[0,0,3] = 1; W2[1,0,3] = 1
    
    # r=4: M5
    U2[0,0,4] = 1; U2[0,1,4] = 1
    V2[1,1,4] = 1
    W2[0,0,4] = -1; W2[0,1,4] = 1
    
    # r=5: M6
    U2[1,0,5] = 1; U2[0,0,5] = -1
    V2[0,0,5] = 1; V2[0,1,5] = 1
    W2[1,1,5] = 1
    
    # r=6: M7
    U2[0,1,6] = 1; U2[1,1,6] = -1
    V2[1,0,6] = 1; V2[1,1,6] = 1
    W2[0,0,6] = 1; W2[1,1,6] = 1
    
    # Now construct 4x4 U, V, W
    # The 4x4 indices i, j can be written as i = 2*i1 + i0, j = 2*j1 + j0
    # The 4x4 Strassen rank is 7*7 = 49.
    # The rank index r is (r1, r2) where r1 is the block index and r2 is the intra-block index.
    # Wait, the recursive structure is:
    # U4[i, j, (r1, r2)] = U2[i1, j1, r1] * U2[i0, j0, r2] ?
    # This is the tensor product.
    
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            U4[i, j, r4] = U2[i1, j1, r1] * U2[i0, j0, r2]
                            
                            for k1 in range(2):
                                for k0 in range(2):
                                    k = 2 * k1 + k0
                                    V4[j, k, r4] = V2[j1, k1, r1] * V2[j0, k0, r2]
                                    W4[k, i, r4] = W2[k1, i1, r1] * W2[k0, i0, r2]
                                    
    # --- Step 2: Add terms for the 5th dimension ---
    # Total rank so far: 49.
    # We need to compute:
    # C[0:4, 4] = A[0:4, 0:4] B[0:4, 4] + A[0:4, 4] B[4, 4]
    # C[4, 0:4] = A[4, 0:4] B[0:4, 0:4] + A[4, 4] B[4, 0:4]
    # C[4, 4] = A[4, 4] B[4, 4]
    
    # Indices: 0..3 are the 4x4 part, 4 is the 5th.
    
    # 1. C[0:4, 4] from A[0:4, 0:4] B[0:4, 4]
    # For each i in 0..3, k=4.
    # C[i, 4] += sum_{j=0}^3 A[i, j] B[j, 4]
    # This requires 4*4 = 16 terms?
    # Term for (i, j): u[i, j]=1, v[j, 4]=1, w[4, i]=1.
    rank_offset = 49
    for i in range(4):
        for j in range(4):
            r = rank_offset
            U4[i, j, r] = 1
            V4[j, 4, r] = 1 # Note: V4 is 4x4, we need to extend to 5x5
            # We will build U, V, W for 5x5 directly.
            
    # Let's restart the construction for 5x5 to avoid index errors.
    
    U = np.zeros((5, 5, 86)) # Max rank estimate
    V = np.zeros((5, 5, 86))
    W = np.zeros((5, 5, 86))
    
    # 1. Copy 4x4 Strassen into the 0:4, 0:4 block
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i, j, r] = U4[i, j, r]
                V[j, r] = V4[j, r] # Wait, V4 is 4x4. V is 5x5.
                # V[j, k, r]
                for k in range(4):
                    V[j, k, r] = V4[j, k, r]
                for k in range(4):
                    W[k, i, r] = W4[k, i, r]
                    
    # 2. Add terms for C[0:4, 4]
    rank = 49
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i, j, rank] = 1
            V[j, 4, rank] = 1
            W[4, i, rank] = 1
            rank += 1
            
    # 3. Add terms for C[0:4, 4] from A[0:4, 4] B[4, 4]
    # C[i, 4] += A[i, 4] B[4, 4]
    for i in range(4):
        U[i, 4, rank] = 1
        V[4, 4, rank] = 1
        W[4, i, rank] = 1
        rank += 1
        
    # 4. Add terms for C[4, 0:4]
    # C[4, k] = sum_{j=0}^3 A[4, j] B[j, k]
    for k in range(4):
        for j in range(4):
            U[4, j, rank] = 1
            V[j, k, rank] = 1
            W[k, 4, rank] = 1
            rank += 1
            
    # 5. Add terms for C[4, 0:4] from A[4, 4] B[4, 0:4]
    # C[4, k] += A[4, 4] B[4, k]
    for k in range(4):
        U[4, 4, rank] = 1
        V[4, k, rank] = 1
        W[k, 4, rank] = 1
        rank += 1
        
    # 6. Add term for C[4, 4]
    # C[4, 4] = A[4, 4] B[4, 4]
    U[4, 4, rank] = 1
    V[4, 4, rank] = 1
    W[4, 4, rank] = 1
    rank += 1
    
    # Total rank used: 49 + 16 + 4 + 16 + 4 + 1 = 90?
    # Let's check the count.
    # 1. 49
    # 2. 16
    # 3. 4
    # 4. 16
    # 5. 4
    # 6. 1
    # Sum: 49 + 16 + 4 + 16 + 4 + 1 = 90.
    
    # Trim the arrays to the actual rank
    U = U[:, :, :rank]
    V = V[:, :, :rank]
    W = W[:, :, :rank]
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
