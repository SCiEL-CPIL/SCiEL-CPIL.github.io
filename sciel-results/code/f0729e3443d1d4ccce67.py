import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 rank 7 decomposition
    # C = A * B
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map this to tensor coefficients U, V, W.
    # For 2x2x2:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    def get_strassen_2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1: (A00+A11)(B00+B11)
        # A00: i=0,j=0. A11: i=1,j=1.
        # B00: j=0,k=0. B11: j=1,k=1.
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C11 -= M1? No, C00 = M1 + ..., C11 = M1 - ...
        # W[k*2+i, c]
        # C00 is i=0,k=0 -> W[0, 0]
        # C11 is i=1,k=1 -> W[3, 0]
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2: (A10+A11)B00
        # A10: i=1,j=0. A11: i=1,j=1.
        # B00: j=0,k=0.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 += M2 (i=1,k=0 -> W[2, 1])
        W[2, 1] = 1
        
        # M3: A00(B01-B11)
        # A00: i=0,j=0.
        # B01: j=0,k=1. B11: j=1,k=1.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += M3 (i=0,k=1 -> W[1, 2])
        W[1, 2] = 1
        
        # M4: A11(B10-B00)
        # A11: i=1,j=1.
        # B10: j=1,k=0. B00: j=0,k=0.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 += M4 (W[0, 3]), C10 += M4 (W[2, 3])
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5: (A00+A01)B11
        # A00: i=0,j=0. A01: i=0,j=1.
        # B11: j=1,k=1.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00 -= M5 (W[0, 4] = -1), C01 += M5 (W[1, 4] = 1)
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6: (A10-A00)(B00+B01)
        # A10: i=1,j=0. A00: i=0,j=0.
        # B00: j=0,k=0. B01: j=0,k=1.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 += M6 (W[3, 5] = 1)
        W[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        # A01: i=0,j=1. A11: i=1,j=1.
        # B10: j=1,k=0. B11: j=1,k=1.
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 += M7 (W[0, 6] = 1), C11 += M7 (W[3, 6] = 1)
        W[0, 6] = 1
        W[3, 6] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    # Construct 4x4x4 rank 28
    # 4 blocks: 00, 01, 10, 11
    # Each block uses the 2x2 decomposition
    # Indices for 4x4: i in 0..3, j in 0..3, k in 0..3
    # Block 00: i in 0,1; j in 0,1; k in 0,1
    # Block 01: i in 0,1; j in 2,3; k in 0,1
    # Block 10: i in 2,3; j in 0,1; k in 0,1
    # Block 11: i in 2,3; j in 2,3; k in 0,1
    
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    
    for block_idx in range(4):
        # i_offset, j_offset, k_offset
        # Block 00: 0,0,0
        # Block 01: 0,2,0
        # Block 10: 2,0,0
        # Block 11: 2,2,0
        # Note: k is always 0..1 for the 2x2 block, but in 4x4, k is 0..3?
        # No, the 4x4 multiplication is C = A B.
        # A is 4x4, B is 4x4.
        # The 2x2 blocks of B are B00, B01, B10, B11.
        # B00 is rows 0,1 cols 0,1.
        # So j (col of A / row of B) is 0,1 for B00.
        # k (col of B) is 0,1 for B00.
        # So for block 00, j in 0,1, k in 0,1.
        # For block 01 (A00 * B01), B01 is rows 0,1 cols 2,3.
        # So j in 0,1, k in 2,3.
        
        i_off = (block_idx // 2) * 2
        j_off = (block_idx % 2) * 2
        k_off = (block_idx % 2) * 2 # Wait, B01 is cols 2,3. B11 is cols 2,3.
        
        # Let's re-verify Strassen blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # The 7 multiplications mix these.
        # M1 = (A00+A11)(B00+B11)
        # This involves A00 (i0, j0), A11 (i1, j1).
        # And B00 (j0, k0), B11 (j1, k1).
        # So M1 is not a single 2x2 block multiplication in the simple sense?
        # No, Strassen computes 7 products of 2x2 matrices.
        # Each product is a rank 7 tensor.
        # The total tensor is the sum of these 7 tensors, mapped to the 4x4 indices.
        
        # My previous loop logic was wrong. I need to map the 2x2 Strassen coefficients to the 4x4 indices for each of the 7 multiplications.
        
        pass # Re-implement properly

    # Correct 4x4 construction
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    
    # The 7 Strassen multiplications M1..M7
    # Each Mc is a 2x2 multiplication of specific linear combinations of blocks.
    # We can just use the 2x2 Strassen tensor for each M and map indices.
    
    # M1: (A00+A11) * (B00+B11)
    # A parts: A00 (i0, j0), A11 (i1, j1)
    # B parts: B00 (j0, k0), B11 (j1, k1)
    # This is NOT a simple 2x2 multiplication of disjoint blocks.
    # It is a multiplication of (A00+A11) and (B00+B11).
    # Let X = A00+A11 (4x4? No, 2x2 block sum).
    # X is a 2x2 matrix of 2x2 blocks? No, X is a 4x4 matrix?
    # No, A00 is 2x2. A11 is 2x2. A00+A11 is 2x2.
    # So M1 is a 2x2 matrix multiplication.
    # The input to this 2x2 multiplication is X (2x2) and Y (2x2).
    # X[0,0] = A00[0,0] + A11[0,0], etc.
    # So the tensor for M1 is the standard 2x2 multiplication tensor, but with U and V modified to sum the blocks.
    
    # This is getting complex to hardcode.
    # I will use the fact that Rank(4x4) = 28 is achieved by 4 independent 2x2 Strassen?
    # NO. 4 * 7 = 28.
    # If I use 4 independent 2x2 Strassen decompositions for the 4 blocks of C?
    # C00 = A00 B00 + A01 B10. This requires 2 multiplications.
    # If I use schoolbook for each block, it's 16 multiplications.
    # Strassen reduces it to 7.
    # The 7 multiplications are of 2x2 matrices.
    # So I need to construct the 4x4 tensor as the sum of 7 2x2 tensors, appropriately embedded.
    
    # I will implement the 7 embeddings.
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (i0,j0), A11 (i1,j1).
    # V: B00 (j0,k0), B11 (j1,k1).
    # W: C00 (i0,k0), C11 (i1,k1).
    
    # I will write a helper to add a 2x2 Strassen tensor to the 4x4 arrays.
    
    def add_strassen_block(U4, V4, W4, offset, u_map, v_map, w_map):
        # u_map: list of (i, j) for the 4 entries of the 2x2 A matrix
        # v_map: list of (j, k) for the 4 entries of the 2x2 B matrix
        # w_map: list of (i, k) for the 4 entries of the 2x2 C matrix
        # offset: column offset in U4, V4, W4
        U2, V2, W2 = get_strassen_2x2()
        for c in range(7):
            for idx in range(4):
                i, j = u_map[idx]
                U4[i*4+j, offset+c] += U2[idx, c]
                j2, k = v_map[idx]
                V4[j2*4+k, offset+c] += V2[idx, c]
                i2, k2 = w_map[idx]
                W4[k2*4+i2, offset+c] += W2[idx, c]

    # M1
    u1 = [(0,0), (0,1), (1,0), (1,1)] # A00
    # Wait, A00+A11.
    # X = A00 + A11.
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    # So u_map for X is not a single block.
    # I need to add contributions from A00 and A11 to the same column.
    
    # This manual mapping is error-prone.
    # I will stick to the Rank 56 solution using two Rank 28 blocks.
    # And I will construct the Rank 28 block using 4 independent Rank 7 blocks?
    # NO, 4 independent Rank 7 blocks give Rank 28 for 4x4?
    # If I compute C00 using A00B00 + A01B10, that's 2 multiplications.
    # If I use Strassen, it's 7.
    # If I use 4 independent 2x2 Strassen, I am computing C00, C01, C10, C11 independently?
    # No, Strassen couples them.
    
    # I will use the Rank 96 schoolbook if I can't get Strassen right quickly.
    # But 56 is much better.
    # I will implement the 4 independent 2x2 Strassen for the 4 output blocks?
    # C00 = A00 B00 + A01 B10.
    # If I just compute A00 B00 and A01 B10 separately, that's 2 * 7 = 14 multiplications for C00.
    # Total 4 * 14 = 56.
    # This is Rank 56 for 4x4.
    # Then Rank 112 for 3x4x8.
    # That's worse than 96.
    
    # I MUST implement the coupled Strassen.
    
    return {"U": [], "V": [], "W": []}

# I will output the schoolbook for now to ensure validity, then try to improve.
# Actually, I can just output the schoolbook.
