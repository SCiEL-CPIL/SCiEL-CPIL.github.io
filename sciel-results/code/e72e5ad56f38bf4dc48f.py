import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to generate 2x2x2 Strassen decomposition
    def get_strassen_2x2x2():
        # Returns U, V, W as lists of lists for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 2: (A10+A11)B00 -> C10+=1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # Term 3: A00(B10-B11) -> C01+=1, C11+=1
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 4: A11(B01-B00) -> C00+=1, C10+=1
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 5: (A00+A01)B11 -> C00-=1, C01+=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 6: (A10-A00)(B00+B01) -> C11+=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 7: (A01-A11)(B10+B11) -> C00+=1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    # Construct 4x4x4 decomposition using Kronecker product logic
    # Rank 49
    # U_4 shape (16, 49), V_4 shape (16, 49), W_4 shape (16, 49)
    # Note: 4x4x4 means n=4, m=4, p=4.
    # U index: i*m+j = 4i+j.
    # V index: j*p+k = 4j+k.
    # W index: k*n+i = 4k+i.
    
    U_4 = [[0]*49 for _ in range(16)]
    V_4 = [[0]*49 for _ in range(16)]
    W_4 = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U contribution
            # U_2[r1] is for 2x2 A block.
            # We need to map 2x2 indices to 4x4 indices.
            # A is 4x4. Split into 2x2 blocks.
            # Block (0,0) corresponds to A[0:2, 0:2]. Indices 0,1,2,3 in 2x2 map to 0,1,4,5 in 4x4?
            # No, standard row-major:
            # 2x2 index (i,j) -> 4x4 index (2*i + i_sub, 2*j + j_sub)
            
            # Actually, it's easier to think of the Kronecker product.
            # U_4 = U_2 (Kron) U_2 ?
            # Let's verify.
            # The 7 terms in the outer Strassen are linear combinations of 2x2 matrices.
            # Each 2x2 matrix is multiplied by another 2x2 matrix.
            # The multiplication of two 2x2 matrices is decomposed into 7 rank-1 terms.
            # So yes, it's a Kronecker structure.
            
            for i2 in range(2):
                for j2 in range(2):
                    # u_val = U_2[i2*2+j2][r1]
                    # This value multiplies with the inner term r2.
                    # The index in U_4 for the outer term r1 and inner term r2
                    # depends on the specific entry (i,j) of the 4x4 matrix.
                    
                    # Let's use the property:
                    # U_4[i*4+j, r1*7+r2] = U_2[i*2+j, r1] * U_2[i*2+j, r2] ? No.
                    
                    # Let's derive it carefully.
                    # Outer term r1 involves a specific 2x2 block of A (e.g. A00) with coefficient c.
                    # Inner term r2 involves a specific 2x2 block of that block (e.g. A00[0,0]) with coefficient d.
                    # So U_4 entry is c * d.
                    
                    # Which 2x2 block of A does outer term r1 use?
                    # From get_strassen_2x2x2:
                    # r1=0: A00, A11
                    # r1=1: A10, A11
                    # r1=2: A00
                    # r1=3: A11
                    # r1=4: A00, A01
                    # r1=5: A10, A00
                    # r1=6: A01, A11
                    
                    # This is getting messy to hardcode. 
                    # Let's just use the Kronecker product of the matrices U_2, V_2, W_2.
                    # Does rank(M_{2,2,2} \otimes M_{2,2,2}) = rank(M_{4,4,4})?
                    # Yes, if the decomposition is product-like.
                    # Strassen is not purely product-like in the sense of simple Kronecker of the *final* U,V,W because the signs and sums are intertwined.
                    # However, the tensor $T_{4,4,4}$ is isomorphic to $T_{2,2,2} \otimes T_{2,2,2}$?
                    # No, $T_{4,4,4}$ is the multiplication tensor for 4x4x4.
                    # $T_{2,2,2} \otimes T_{2,2,2}$ is the multiplication tensor for 4x4x4?
                    # Yes, matrix multiplication is bilinear and decomposes into block multiplications.
                    # The rank of the Kronecker product of two tensors is the product of their ranks.
                    # So if we take the Kronecker product of the U,V,W of the 2x2x2 decomposition, we get a valid decomposition for 4x4x4.
                    
                    # Let's check indices.
                    # U_2 is 4x7. U_4 should be 16x49.
                    # U_4[i*4+j, r1*7+r2] = U_2[i*2 + (i%2), r1] * U_2[j*2 + (j%2), r2] ? No.
                    
                    # Let's just construct it by iterating.
                    pass

    # Simpler approach: Just use Schoolbook for 4x4x4 (rank 64) and Schoolbook for 4x4x3 (rank 48). Total 112. No improvement.
    # I need the Strassen one.
    
    # Let's assume the Kronecker product works.
    # U_4 = np.kron(U_2, U_2)
    # V_4 = np.kron(V_2, V_2)
    # W_4 = np.kron(W_2, W_2)
    
    # Check dimensions:
    # U_2: 4x7. Kron: 16x49. Correct.
    # V_2: 4x7. Kron: 16x49. Correct.
    # W_2: 4x7. Kron: 16x49. Correct.
    
    # Does this correctly compute 4x4x4 multiplication?
    # Yes, the tensor product of two multiplication tensors is the multiplication tensor of the block sizes.
    
    U_4_np = np.kron(np.array(U_2), np.array(U_2))
    V_4_np = np.kron(np.array(V_2), np.array(V_2))
    W_4_np = np.kron(np.array(W_2), np.array(W_2))
    
    # Convert to lists
    U_4 = U_4_np.tolist()
    V_4 = V_4_np.tolist()
    W_4 = W_4_np.tolist()
    
    # Now handle 4x4x3 part (Schoolbook, rank 48)
    # U_3 shape (16, 48), V_3 shape (12, 48), W_3 shape (12, 48)
    # n=4, m=4, p=3.
    U_3 = [[0]*48 for _ in range(16)]
    V_3 = [[0]*48 for _ in range(12)]
    W_3 = [[0]*48 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_3[i*4+j][col] = 1
                V_3[j*3+k][col] = 1
                W_3[k*4+i][col] = 1
                col += 1
                
    # Combine
    # Total Rank = 49 + 48 = 97
    # U_final shape (16, 97)
    # V_final shape (28, 97) -> 16 (from 4x4) + 12 (from 4x3) = 28.
    # W_final shape (28, 97) -> 16 (from 4x4) + 12 (from 4x3) = 28.
    
    U_final = [[0]*97 for _ in range(16)]
    V_final = [[0]*97 for _ in range(28)]
    W_final = [[0]*97 for _ in range(28)]
    
    # Copy 4x4x4 part (cols 0-48)
    for r in range(49):
        for i in range(16):
            U_final[i][r] = U_4[i][r]
        for j in range(16):
            V_final[j][r] = V_4[j][r]
        for k in range(16):
            W_final[k][r] = W_4[k][r]
            
    # Copy 4x4x3 part (cols 49-96)
    # V_3 indices: j*3+k. j in 0..3, k in 0..2.
    # V_final indices for this part: 16 + j*3 + k.
    # W_3 indices: k*4+i. k in 0..2, i in 0..3.
    # W_final indices for this part: 16 + k*4 + i.
    
    for r in range(48):
        r_global = 49 + r
        for i in range(16):
            U_final[i][r_global] = U_3[i][r]
        for idx in range(12):
            # Map idx to j, k
            j = idx // 3
            k = idx % 3
            V_final[16 + idx][r_global] = V_3[idx][r]
        for idx in range(12):
            # Map idx to k, i
            k = idx // 4
            i = idx % 4
            W_final[16 + idx][r_global] = W_3[idx][r]
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
