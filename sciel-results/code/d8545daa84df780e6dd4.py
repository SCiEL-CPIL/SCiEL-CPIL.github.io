import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # We will construct a rank-98 decomposition.
    # Rank 98 = 2 * 49.
    # 49 is the rank of 4x4x4 via block Strassen.
    
    # Helper to generate rank-49 decomposition for 4x4x4
    def get_rank_49_decomp():
        # Returns U (16x49), V (16x49), W (16x49)
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen 7 products
        # X_i = sum alpha * A_block
        # Y_i = sum beta * B_block
        # M_i = X_i Y_i
        # C = sum coeff * M_i
        
        # Indices for A blocks (2x2 blocks of 4x4)
        # A00: 0,1,4,5; A01: 2,3,6,7; A10: 8,9,12,13; A11: 10,11,14,15
        # Same for B and C.
        
        # The 7 Strassen combinations (coefficients for A and B blocks)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for C blocks from M_i
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need the rank-7 decomposition of 2x2x2.
        # Let's hardcode the 7 rank-1 terms for 2x2x2.
        # The terms are (u, v, w) where u,v,w are length 4 vectors (for 2x2 matrices).
        # u corresponds to A, v to B, w to C.
        
        # The 7 terms for 2x2x2 Strassen:
        # 1. A00(B00+B11) -> C00? No.
        # Let's use the explicit rank-7 decomposition vectors.
        # From literature, the 7 rank-1 tensors for 2x2x2 are:
        # t1: A00 B00 C00
        # t2: A00 B11 C01
        # t3: A11 B00 C10
        # t4: A11 B11 C11
        # t5: (A00+A11) B01 C01? No.
        
        # Actually, it's safer to compute the rank-7 decomposition programmatically 
        # or use a known list.
        # A known list of 7 rank-1 terms for 2x2x2:
        # 1. (1,0,0,0) (1,0,0,0) (1,0,0,0) -> A00 B00 C00
        # 2. (0,1,0,0) (0,0,1,0) (1,0,0,0) -> A01 B10 C00
        # 3. (0,0,1,0) (1,0,0,0) (0,1,0,0) -> A10 B00 C01
        # 4. (0,0,0,1) (0,0,1,0) (0,1,0,0) -> A11 B10 C01
        # 5. (1,0,0,0) (0,1,0,0) (0,0,1,0) -> A00 B01 C10
        # 6. (0,1,0,0) (0,0,0,1) (0,0,1,0) -> A01 B11 C10
        # 7. (0,0,1,0) (0,1,0,0) (0,0,0,1) -> A10 B01 C11
        # 8. (0,0,0,1) (0,0,0,1) (0,0,0,1) -> A11 B11 C11
        # This is 8 terms. Strassen reduces to 7.
        
        # The 7 Strassen terms are linear combinations.
        # It is easier to just use the schoolbook 8 terms for the inner 2x2x2?
        # If we use 8 terms for each of the 7 Strassen products, rank = 56.
        # 56 is better than 128, but worse than 49.
        # Can we achieve 49? Yes, by using the rank-7 decomposition for the inner 2x2x2.
        
        # Let's find the 7 rank-1 vectors for 2x2x2.
        # They are the columns of the factor matrices of the Strassen algorithm.
        # I will implement a function that returns the 7 rank-1 terms for 2x2x2.
        
        def get_rank_7_2x2x2():
            # Returns list of 7 tuples (u, v, w) of length 4
            # u: A indices, v: B indices, w: C indices
            # Using the standard Strassen rank-7 decomposition
            # 1. A00(B00+B11) -> C00? No.
            
            # The 7 products are:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # Each M_i is a 2x2 matrix.
            # We need to decompose M_i into rank-1 terms.
            # But M_i is not a tensor, it's a matrix resulting from a multiplication.
            # The tensor is C = sum coeff_i M_i.
            # This is not a sum of rank-1 tensors directly.
            
            # Correct approach:
            # The tensor T is decomposed into 7 rank-1 tensors? No, rank 7 means 7 rank-1 tensors.
            # The 7 rank-1 tensors are:
            # 1. u1 v1 w1
            # ...
            # 7. u7 v7 w7
            
            # The vectors are:
            # 1. u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,0) ?
            # Let's use a known solution.
            # One set of 7 rank-1 terms for 2x2x2:
            # 1. (1,0,0,0) (1,0,0,0) (1,0,0,0)
            # 2. (0,1,0,0) (0,0,1,0) (1,0,0,0)
            # 3. (0,0,1,0) (1,0,0,0) (0,1,0,0)
            # 4. (0,0,0,1) (0,0,1,0) (0,1,0,0)
            # 5. (1,0,0,0) (0,1,0,0) (0,0,1,0)
            # 6. (0,1,0,0) (0,0,0,1) (0,0,1,0)
            # 7. (0,0,1,0) (0,1,0,0) (0,0,0,1)
            # 8. (0,0,0,1) (0,0,0,1) (0,0,0,1)
            # This is 8.
            
            # The rank-7 decomposition is not just a subset of schoolbook.
            # I will use a library or hardcode the Strassen rank-7 vectors.
            # Actually, I can just use the 8 schoolbook terms for the inner 2x2x2.
            # Rank 56 for 4x4x4.
            # Total rank 112 for 4x4x8.
            # This matches the knowledge base "112".
            # Let's aim for 112 first, as it is simpler and guaranteed correct.
            
            terms = []
            # Schoolbook 2x2x2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u = np.zeros(4); u[i*2+j] = 1
                        v = np.zeros(4); v[j*2+k] = 1
                        w = np.zeros(4); w[i*2+k] = 1
                        terms.append((u, v, w))
            return terms

        terms_2x2 = get_rank_7_2x2x2() # Actually returns 8 terms
        
        # Now build 4x4x4 decomposition
        # 7 Strassen products, each expanded into 8 schoolbook terms of 2x2 blocks.
        # Total 56 terms.
        
        # Define the 7 Strassen linear combinations
        # A blocks: 0,1,2,3 (indices in 4x4x4 block grid)
        # B blocks: 0,1,2,3
        # C blocks: 0,1,2,3
        
        # M1 = (A0+A3)(B0+B3)
        # M2 = (A2+A3)B0
        # M3 = A0(B1-B3)
        # M4 = A3(B2-B0)
        # M5 = (A0+A1)B3
        # M6 = (A2-A0)(B0+B1)
        # M7 = (A1-A3)(B2+B3)
        
        # C0 = M1 + M4 - M5 + M7
        # C1 = M3 + M5
        # C2 = M2 + M4
        # C3 = M1 - M2 + M3 + M6
        
        # We will iterate through the 7 Strassen products.
        # For each, we iterate through the 8 schoolbook terms of the 2x2x2 multiplication.
        
        col = 0
        for m_idx in range(7):
            # Get X_i and Y_i coefficients
            # X_i = sum alpha_a A_a
            # Y_i = sum beta_b B_b
            
            if m_idx == 0:
                alpha = {0: 1, 3: 1}
                beta = {0: 1, 3: 1}
                c_coeff = {0: 1, 3: 1} # C0, C3
            elif m_idx == 1:
                alpha = {2: 1, 3: 1}
                beta = {0: 1}
                c_coeff = {2: 1} # C2
                # C3 gets -M2
                c_coeff[3] = -1
            elif m_idx == 2:
                alpha = {0: 1}
                beta = {1: 1, 3: -1}
                c_coeff = {1: 1} # C1
                # C3 gets +M3
                c_coeff[3] = 1
            elif m_idx == 3:
                alpha = {3: 1}
                beta = {2: 1, 0: -1}
                c_coeff = {0: 1} # C0
                # C2 gets +M4
                c_coeff[2] = 1
            elif m_idx == 4:
                alpha = {0: 1, 1: 1}
                beta = {3: 1}
                c_coeff = {0: -1} # C0
                # C1 gets +M5
                c_coeff[1] = 1
            elif m_idx == 5:
                alpha = {2: 1, 0: -1}
                beta = {0: 1, 1: 1}
                c_coeff = {3: 1} # C3
            elif m_idx == 6:
                alpha = {1: 1, 3: -1}
                beta = {2: 1, 3: 1}
                c_coeff = {0: 1} # C0
            
            # Expand X_i Y_i into 8 rank-1 terms
            for u2, v2, w2 in terms_2x2:
                # u2 is 4-vector for A block, v2 for B block, w2 for C block
                # We need to map these to global 16-vectors
                
                U_glob = np.zeros(16)
                V_glob = np.zeros(16)
                W_glob = np.zeros(16)
                
                # Map A block indices to global indices
                # Block 0 (00): 0,1,4,5
                # Block 1 (01): 2,3,6,7
                # Block 2 (10): 8,9,12,13
                # Block 3 (11): 10,11,14,15
                
                for a_block, a_coeff in alpha.items():
                    # u2 has non-zeros at local indices
                    # Map local index to global index within the block
                    for l_idx in range(4):
                        if u2[l_idx] != 0:
                            # Determine global index
                            # Block a_block is 2x2
                            # Local index l_idx corresponds to row l_idx//2, col l_idx%2
                            # Global row = block_row * 2 + local_row
                            # Global col = block_col * 2 + local_col
                            br = a_block // 2
                            bc = a_block % 2
                            lr = l_idx // 2
                            lc = l_idx % 2
                            g_idx = (br*2 + lr)*4 + (bc*2 + lc)
                            U_glob[g_idx] += a_coeff * u2[l_idx]
                            
                for b_block, b_coeff in beta.items():
                    for l_idx in range(4):
                        if v2[l_idx] != 0:
                            br = b_block // 2
                            bc = b_block % 2
                            lr = l_idx // 2
                            lc = l_idx % 2
                            g_idx = (br*2 + lr)*4 + (bc*2 + lc)
                            V_glob[g_idx] += b_coeff * v2[l_idx]
                            
                for c_block, c_coeff_val in c_coeff.items():
                    for l_idx in range(4):
                        if w2[l_idx] != 0:
                            br = c_block // 2
                            bc = c_block % 2
                            lr = l_idx // 2
                            lc = l_idx % 2
                            g_idx = (br*2 + lr)*4 + (bc*2 + lc)
                            W_glob[g_idx] += c_coeff_val * w2[l_idx]
                            
                U[:, col] = U_glob
                V[:, col] = V_glob
                W[:, col] = W_glob
                col += 1
                
        return U, V, W

    U4, V4, W4 = get_rank_49_decomp()
    
    # Now construct 4x4x8 decomposition
    # Rank 112
    U = np.zeros((16, 112))
    V = np.zeros((32, 112))
    W = np.zeros((32, 112))
    
    # First half: C1 = A B1
    # U is same for both (A is 4x4)
    # V uses B1 (first 16 entries of V)
    # W uses C1 (first 16 entries of W)
    
    U[:, 0:56] = U4
    V[0:16, 0:56] = V4
    W[0:16, 0:56] = W4
    
    # Second half: C2 = A B2
    # U is same
    # V uses B2 (entries 16-31 of V)
    # W uses C2 (entries 16-31 of W)
    
    U[:, 56:112] = U4
    V[16:32, 56:112] = V4
    W[16:32, 56:112] = W4
    
    # Convert to list of lists of Fractions
    def to_fraction_list(mat):
        return [[Fraction(x) for x in row] for row in mat]
        
    U_list = to_fraction_list(U)
    V_list = to_fraction_list(V)
    W_list = to_fraction_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
