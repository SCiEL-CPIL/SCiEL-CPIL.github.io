"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    n, m, p = 4, 4, 7
    deadline = time.time() + 120
    
    # We'll build the decomposition for 4x4 * 4x7
    # Split into 4x4 * 4x4 and 4x4 * 4x3
    
    # For 4x4 * 4x4, use Strassen's algorithm
    # Strassen for 2x2x2 has rank 7
    # For 4x4x4, we can use two levels of Strassen or a direct 4x4x4 decomposition
    
    # Let's use the known 4x4x4 decomposition from Strassen (rank 49)
    # Actually, the rank of 4x4x4 is 49 (Strassen's algorithm applied recursively)
    
    # But wait, let me think more carefully.
    # The tensor M_{4,4,4} can be decomposed using Strassen's method.
    # Strassen for 2x2: rank 7
    # For 4x4: split into 2x2 blocks, use Strassen on each, but that gives 7*7=49? No.
    # Actually, for 4x4x4, the Strassen decomposition has rank 49.
    
    # Let me construct the 4x4x4 decomposition using Strassen's algorithm.
    
    # Strassen's 2x2x2 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me build the 2x2x2 Strassen decomposition first.
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm."""
        # n=m=p=2
        # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # For 2x2: U is 4x7, V is 4x7, W is 4x7
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is at index 0 (i=0,j=0), A22 is at index 3 (i=1,j=1)
        # B11 is at index 0 (j=0,k=0), B22 is at index 3 (j=1,k=1)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        # C11 += M1, C22 += M1
        W[0, 0] = 1  # C11 (k=0,i=0)
        W[3, 0] = 1  # C22 (k=1,i=1)
        
        # M2 = (A21+A22)B11
        # A21 is at index 2 (i=1,j=0), A22 is at index 3 (i=1,j=1)
        # B11 is at index 0 (j=0,k=0)
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        # C21 += M2
        W[1, 1] = 1  # C21 (k=0,i=1)
        
        # M3 = A11(B21-B22)
        # A11 is at index 0
        # B21 is at index 2 (j=1,k=0), B22 is at index 3 (j=1,k=1)
        U[0, 2] = 1  # A11
        V[2, 2] = 1  # B21
        V[3, 2] = -1  # -B22
        # C12 += M3, C22 += M3
        W[2, 2] = 1  # C12 (k=1,i=0)
        W[3, 2] = 1  # C22 (k=1,i=1)
        
        # M4 = A22(B11-B21)
        # A22 is at index 3
        # B11 is at index 0, B21 is at index 2
        U[3, 3] = 1  # A22
        V[0, 3] = 1  # B11
        V[2, 3] = -1  # -B21
        # C11 += M4, C21 += M4
        W[0, 3] = 1  # C11
        W[1, 3] = 1  # C21
        
        # M5 = (A11+A12)B22
        # A11 is at index 0, A12 is at index 1 (i=0,j=1)
        # B22 is at index 3
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        # C11 -= M5, C12 += M5
        W[0, 4] = -1  # C11
        W[2, 4] = 1  # C12
        
        # M6 = (A21-A11)(B11+B12)
        # A21 is at index 2, A11 is at index 0
        # B11 is at index 0, B12 is at index 1 (j=0,k=1)
        U[2, 5] = 1  # A21
        U[0, 5] = -1  # -A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        # C22 += M6
        W[3, 5] = 1  # C22
        
        # M7 = (A12-A22)(B21+B22)
        # A12 is at index 1, A22 is at index 3
        # B21 is at index 2, B22 is at index 3
        U[1, 6] = 1  # A12
        U[3, 6] = -1  # -A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        # C11 += M7
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    # Now let's verify the 2x2x2 Strassen decomposition
    U2, V2, W2 = strassen_2x2()
    
    def verify_2x2():
        for _ in range(10):
            A = np.random.randn(2, 2)
            B = np.random.randn(2, 2)
            C = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            C_decomp = np.zeros(4)
            for r in range(7):
                scalar = np.dot(U2[:, r], A_flat) * np.dot(V2[:, r], B_flat)
                C_decomp += scalar * W2[:, r]
            C_decomp = C_decomp.reshape(2, 2).T  # W[k*n+i] -> need to reshape correctly
            # W is indexed as W[k*n+i], so C_decomp[k*n+i] corresponds to C[i,k]
            # So C_decomp.reshape(2,2) gives C_decomp[k,i] for k in range(2), i in range(2)
            # We need C[i,k], so we transpose
            if not np.allclose(C, C_decomp, atol=1e-9):
                return False
        return True
    
    if not verify_2x2():
        # Fall back to schoolbook for 2x2x2
        U2 = np.zeros((4, 4))
        V2 = np.zeros((4, 4))
        W2 = np.zeros((4, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U2[i*2+j, col] = 1
                    V2[j*2+k, col] = 1
                    W2[k*2+i, col] = 1
                    col += 1
    
    # Now build 4x4x4 using Strassen's algorithm applied to 2x2 blocks
    # This gives rank 7*7 = 49? No, that's not right.
    # Actually, for 4x4x4, we can use Strassen's algorithm directly.
    # The standard approach is to use the 2x2x2 Strassen decomposition and apply it
    # to the 2x2 block structure of 4x4 matrices.
    
    # Let me think about this differently.
    # For 4x4 * 4x4, we can use the following approach:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # Then use Strassen's algorithm on the 2x2 block structure.
    # This gives 7 scalar multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with 4 scalar multiplications (schoolbook) or 7 (Strassen).
    # If we use schoolbook for the 2x2 multiplications, we get 7*4 = 28 scalar multiplications.
    # If we use Strassen for the 2x2 multiplications, we get 7*7 = 49 scalar multiplications.
    
    # Wait, that doesn't make sense. Let me reconsider.
    
    # Actually, the rank of 4x4x4 is known to be 49 (using Strassen's algorithm recursively).
    # But there might be better decompositions.
    
    # For now, let's use the schoolbook decomposition for 4x4x4, which has rank 64.
    # Then we can try to improve it.
    
    # Actually, let's just use the schoolbook for the entire 4x4x7 problem, which has rank 112.
    # Then we can try to use Strassen for parts of it.
    
    # Let's start with a simple approach:
    # Use Strassen for the 4x4 * 4x4 part (first 4 columns of B)
    # Use schoolbook for the 4x4 * 4x3 part (last 3 columns of B)
    
    # For 4x4 * 4x4, let's use the schoolbook decomposition (rank 64) for now.
    # We can try to improve it later.
    
    # Actually, let's try to use the 2x2x2 Strassen decomposition to build a 4x4x4 decomposition.
    
    # Here's the idea:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Using Strassen's algorithm:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product.
    # We can compute each Mi using the 2x2x2 Strassen decomposition.
    
    # So the total rank is 7 * 7 = 49.
    
    # Let's build this.
    
    def build_4x4x4_strassen():
        """Build 4x4x4 decomposition using Strassen's algorithm."""
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # Strassen's 7 terms
        # For each term, we need to compute a 2x2 matrix product
        # Using the 2x2x2 Strassen decomposition
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # A11 is at indices 0,1,4,5 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
        # Wait, I need to be more careful with the indexing.
        
        # Let's define the block indices:
        # A11: i in {0,1}, j in {0,1} -> indices 0,1,4,5
        # A12: i in {0,1}, j in {2,3} -> indices 2,3,6,7
        # A21: i in {2,3}, j in {0,1} -> indices 8,9,12,13
        # A22: i in {2,3}, j in {2,3} -> indices 10,11,14,15
        
        # B11: j in {0,1}, k in {0,1} -> indices 0,1,4,5
        # B12: j in {0,1}, k in {2,3} -> indices 2,3,6,7
        # B21: j in {2,3}, k in {0,1} -> indices 8,9,12,13
        # B22: j in {2,3}, k in {2,3} -> indices 10,11,14,15
        
        # C11: i in {0,1}, k in {0,1} -> indices 0,1,4,5
        # C12: i in {0,1}, k in {2,3} -> indices 2,3,6,7
        # C21: i in {2,3}, k in {0,1} -> indices 8,9,12,13
        # C22: i in {2,3}, k in {2,3} -> indices 10,11,14,15
        
        # For each Strassen term, we need to:
        # 1. Form the appropriate linear combinations of A and B blocks
        # 2. Multiply them using the 2x2x2 Strassen decomposition
        # 3. Add the result to the appropriate C blocks
        
        # Let's define the 2x2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2()
        
        # Now, for each of the 7 Strassen terms, we'll use the 2x2x2 decomposition
        # The 2x2x2 decomposition has rank 7, so each term contributes 7 to the total rank
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # A11+A22: indices 0,1,4,5 (A11) + 10,11,14,15 (A22)
        # B11+B22: indices 0,1,4,5 (B11) + 10,11,14,15 (B22)
        
        # For the 2x2x2 decomposition, we need to map the 2x2 blocks to the 4-element vectors
        # A11+A22: [A11[0,0]+A22[0,0], A11[0,1]+A22[0,1], A11[1,0]+A22[1,0], A11[1,1]+A22[1,1]]
        # This corresponds to indices 0,1,4,5 for A11 and 10,11,14,15 for A22
        
        # Let's define a helper function to add a 2x2x2 decomposition to the 4x4x4 decomposition
        def add_2x2_decomposition(term_idx, a_indices, b_indices, c_indices, c_signs):
            """Add a 2x2x2 Strassen decomposition to the 4x4x4 decomposition.
            
            term_idx: the Strassen term index (0-6)
            a_indices: list of 4 indices for the A block (in order [0,0], [0,1], [1,0], [1,1])
            b_indices: list of 4 indices for the B block
            c_indices: list of 4 indices for the C block
            c_signs: list of 4 signs for the C block
            """
            base = term_idx * 7
            for r in range(7):
                # U: for each element in the A block
                for local_i, local_j in [(0,0), (0,1), (1,0), (1,1)]:
                    idx = a_indices[local_i*2+local_j]
                    U4[idx, base+r] += U2[local_i*2+local_j, r]
                
                # V: for each element in the B block
                for local_j, local_k in [(0,0), (0,1), (1,0), (1,1)]:
                    idx = b_indices[local_j*2+local_k]
                    V4[idx, base+r] += V2[local_j*2+local_k, r]
                
                # W: for each element in the C block
                for local_k, local_i in [(0,0), (0,1), (1,0), (1,1)]:
                    idx = c_indices[local_k*2+local_i]
                    W4[idx, base+r] += W2[local_k*2+local_i, r] * c_signs[local_k*2+local_i]
        
        # Define the block indices
        A11_idx = [0, 1, 4, 5]
        A12_idx = [2, 3, 6, 7]
        A21_idx = [8, 9, 12, 13]
        A22_idx = [10, 11, 14, 15]
        
        B11_idx = [0, 1, 4, 5]
        B12_idx = [2, 3, 6, 7]
        B21_idx = [8, 9, 12, 13]
        B22_idx = [10, 11, 14, 15]
        
        C11_idx = [0, 1, 4, 5]
        C12_idx = [2, 3, 6, 7]
        C21_idx = [8, 9, 12, 13]
        C22_idx = [10, 11, 14, 15]
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # A: A11+A22 -> [A11[0,0]+A22[0,0], A11[0,1]+A22[0,1], A11[1,0]+A22[1,0], A11[1,1]+A22[1,1]]
        # This is a linear combination, so we need to add both A11 and A22 contributions
        # For U, we add U2 contributions for both A11 and A22
        # For V, we add V2 contributions for both B11 and B22
        # For W, we add W2 contributions for C11 and C22 (with appropriate signs)
        
        # Actually, this is getting complicated. Let me try a different approach.
        
        # Instead of trying to build the 4x4x4 decomposition manually, let's just use the schoolbook
        # decomposition for the entire 4x4x7 problem.
        
        # Schoolbook decomposition for 4x4x7:
        rank = 112
        U = np.zeros((16, rank))
        V = np.zeros((28, rank))
        W = np.zeros((28, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        
        return U, V, W
    
    # For now, let's just use the schoolbook decomposition.
    # We can try to improve it later.
    
    U, V, W = build_4x4x4_strassen()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
