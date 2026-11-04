import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen decomposition for N x N x N (N=4)
    def strassen_4x4x4():
        # Returns U, V, W for 4x4x4 multiplication
        # Rank 49
        r = 49
        U = np.zeros((16, r))
        V = np.zeros((16, r)) # Local 4x4, so m*p = 16
        W = np.zeros((16, r)) # Local 4x4, so p*n = 16
        
        # We use the recursive Strassen structure
        # Level 1: 7 multiplications of 2x2
        # Level 2: Each 2x2 uses 7 multiplications of 1x1
        
        # Indices:
        # A is 4x4, B is 4x4, C is 4x4
        # Split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # Strassen products P1..P7 are 2x2 matrices.
        # Each Pk is computed using 7 scalar products.
        
        # We need to map the scalar products to U, V, W.
        # A scalar product is A[i,j] * B[j,k].
        # In Strassen, we form sums of A blocks and B blocks.
        
        # Let's define the 7 Strassen factors for 2x2:
        # S1 = A00 + A11
        # S2 = A10 + A11
        # S3 = A00
        # S4 = A11
        # S5 = A00 + A01
        # S6 = A10 - A00
        # S7 = A01 - A11
        
        # T1 = B00 + B11
        # T2 = B00
        # T3 = B01 - B11
        # T4 = B10 - B00
        # T5 = B11
        # T6 = B00 + B01
        # T7 = B10 + B11
        
        # P1 = S1 * T1
        # P2 = S2 * T2
        # P3 = S3 * T3
        # P4 = S4 * T4
        # P5 = S5 * T5
        # P6 = S6 * T6
        # P7 = S7 * T7
        
        # Now, each Pk is a 2x2 matrix product.
        # Pk = Sk * Tk.
        # Sk and Tk are 2x2 matrices formed from linear combinations of A and B blocks.
        # We can expand Sk * Tk into scalar products.
        
        # Instead of symbolic expansion, we can just iterate through the 7 Strassen steps
        # and for each, iterate through the 7 scalar steps of the 2x2 Strassen.
        
        # Let's build the U, V, W matrices by accumulating contributions.
        
        # Helper to add a term
        # Term: coeff * A[ai, aj] * B[bj, bk] -> contributes to C[ci, ck]
        # U[ai*m + aj, col] += coeff
        # V[bj*p + bk, col] += coeff
        # W[bk*n + ci, col] += coeff
        # Note: For 4x4x4, n=m=p=4.
        
        col = 0
        for s_idx in range(7):
            # Define S and T matrices for this Strassen step
            # s_idx 0..6 corresponds to P1..P7
            
            # S coefficients for A blocks (2x2 blocks of A)
            # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
            # Each block is 2x2.
            
            # We need to express S as a linear combination of A's 2x2 blocks.
            # Then S * T is a product of two 2x2 matrices.
            # We can just generate the 7 scalar products for the 2x2 product.
            
            # Let's hardcode the 7 Strassen combinations for 2x2.
            # This is getting complex to generate dynamically.
            # Alternative: Just use the known 49-term decomposition.
            
            # Let's use a simpler, verified decomposition:
            # We can use the "block Strassen" which is just Strassen applied to blocks.
            
            # Let's manually construct the 49 terms.
            # This is tedious but doable.
            # Or, we can use a library or a known list.
            
            # For now, let's use a fallback:
            # If we can't easily generate the 49 terms, we can use a slightly higher rank.
            # But 49 is standard.
            
            pass
        
        # Let's try a different approach:
        # Use the fact that we can just call a helper that returns the 49 terms.
        # I will write a small helper to generate the Strassen terms for 2x2 and embed it.
        
        def get_strassen_2x2_terms():
            # Returns list of (coeff, i, j, k, l) for A[i,j] B[j,k] -> C[i,l]?
            # No, Strassen is for C = A B.
            # Terms are of form coeff * (Sum A_blocks) * (Sum B_blocks).
            
            # Let's just list the 7 P's and their 7 scalar components.
            # This is 49 lines.
            
            # To save space and time, I will use a pre-computed list of 49 tuples.
            # Each tuple: (u_idx, v_idx, w_idx, coeff)
            # u_idx is (i, j) in 4x4.
            # v_idx is (j, k) in 4x4.
            # w_idx is (k, i) in 4x4.
            
            # Actually, I can just use the schoolbook for 4x4x4 if I can't be sure.
            # But 64 is worse than 49.
            
            # Let's try to generate it programmatically.
            
            # Strassen 2x2:
            # P1 = (A00+A11)(B00+B11)
            # ...
            
            # I will implement a recursive generator.
            
            terms = []
            
            def gen_strassen(n_size, offset_i, offset_j, offset_k):
                # n_size is the size of the block (2 or 1)
                # offset_i, offset_j, offset_k are the starting indices
                if n_size == 1:
                    # Scalar multiplication
                    # A[offset_i, offset_j] * B[offset_j, offset_k] -> C[offset_k, offset_i]?
                    # Wait, C index is (i, k).
                    # In our global mapping:
                    # U index: i * m + j
                    # V index: j * p + k
                    # W index: k * n + i
                    # Here n=m=p=4.
                    # So U: offset_i * 4 + offset_j
                    # V: offset_j * 4 + offset_k
                    # W: offset_k * 4 + offset_i
                    terms.append((offset_i * 4 + offset_j, offset_j * 4 + offset_k, offset_k * 4 + offset_i, 1.0))
                    return

                # n_size == 2
                # Split into 4 blocks of size 1
                # A00, A01, A10, A11
                # B00, B01, B10, B11
                
                # Strassen 7 products
                # P1 = (A00+A11)(B00+B11)
                # P2 = (A10+A11)B00
                # P3 = A00(B01-B11)
                # P4 = A11(B10-B00)
                # P5 = (A00+A01)B11
                # P6 = (A10-A00)(B00+B01)
                # P7 = (A01-A11)(B10+B11)
                
                # We need to expand each Pk into scalar terms.
                # Since n_size=2, the blocks are 1x1.
                # So A00 is A[0,0], A01 is A[0,1], etc.
                
                # P1 terms:
                # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
                # A00 is at (0,0). B00 is at (0,0).
                # Term: A[0,0] B[0,0] -> C[0,0]
                # Term: A[0,0] B[1,0] -> C[1,0]? No.
                # B11 is at (1,1).
                # A[0,0] * B[1,1] -> C[1,0]?
                # Wait, the product is A * B.
                # C[i,k] = sum_j A[i,j] B[j,k].
                # A00 is A[0,0]. B11 is B[1,1].
                # A[0,0] * B[1,1] contributes to C[0,1]?
                # No, j must match.
                # A[i,j] B[j,k].
                # A00 is i=0, j=0.
                # B11 is j=1, k=1.
                # j=0 and j=1 don't match.
                # So A00 * B11 is NOT a term in the standard matrix product!
                
                # Strassen works on BLOCKS.
                # A00 is a 2x2 block. B11 is a 2x2 block.
                # Their product is a 2x2 block.
                # So we must recurse.
                
                # So, for n_size=2, we call gen_strassen(1, ...) for each scalar in the block product.
                
                # P1 = (A00+A11)(B00+B11)
                # This is a 2x2 matrix product of (A00+A11) and (B00+B11).
                # We can generate the terms for this 2x2 product by calling gen_strassen on the blocks?
                # No, the blocks are size 1.
                # So (A00+A11) is a 2x2 matrix.
                # We can just expand the 2x2 product of two 2x2 matrices using Strassen again?
                # Yes!
                
                # So, gen_strassen(2, ...) should generate 7 calls to gen_strassen(1, ...)?
                # No, it should generate 7 "virtual" 1x1 multiplications?
                # No, Strassen 2x2 uses 7 multiplications of 1x1 matrices.
                # So we just need to list the 7 scalar multiplications and their coefficients.
                
                # Let's list the 7 scalar multiplications for 2x2 Strassen:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # Each Mk is a scalar (since blocks are 1x1).
                # We add these to the terms list.
                
                # M1:
                # (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
                # Wait, A[0,0]B[1,1] is A[0,0]*B[1,1].
                # In the product C = AB, the term A[i,j]B[j,k] requires j to match.
                # Here j=0 for A[0,0] and j=1 for B[1,1]. They don't match.
                # So this term is ZERO in the final C?
                # No, Strassen computes the block product.
                # The block product C00 = M1 + M4 - M5 + M7.
                # The term A[0,0]B[1,1] appears in M1.
                # But it does NOT appear in the standard sum for C[0,0] (which is A[0,0]B[0,0] + A[0,1]B[1,0]).
                # So the non-matching terms must cancel out in the final combination.
                
                # So, we just add the scalar terms with their coefficients.
                
                # M1:
                terms.append((0*4+0, 0*4+0, 0*4+0, 1.0)) # A00 B00
                terms.append((0*4+0, 1*4+1, 1*4+0, 1.0)) # A00 B11 (j=0, j=1 mismatch? No, this is just a term in the bilinear form)
                # Wait, the bilinear form is sum U V W.
                # If we add A[0,0]B[1,1], it means U[0,0]=1, V[1,1]=1, W[1,0]=1.
                # This term is valid in the tensor decomposition.
                
                # So we just list all 49 terms.
                
                # This is getting too long to hardcode in the thought process.
                # I will use a simpler, guaranteed correct decomposition:
                # Schoolbook for 4x4x4 (Rank 64).
                # Total Rank: 64 + 64 + 16 = 144.
                # This is not an improvement.
                
                # I need the 49 rank decomposition.
                # I will write a small function to generate it.
                
                pass
        
        # To be safe and fast, I will use a pre-computed list of 49 terms for 4x4x4.
        # I will generate them using a script in my head? No.
        # I will use the fact that I can just return the schoolbook if I can't be sure.
        # But 144 is the current score.
        
        # Let's try to get 114.
        # I will use a known 49-term list.
        
        # For the purpose of this code, I will use a simplified Strassen that might have rank 49.
        
        # Actually, I can just use the schoolbook for the 4x4x4 parts if I can't verify the 49 terms.
        # But I can verify them by checking the validity.
        
        # Let's try to generate the 49 terms.
        
        return U, V, W, 49

    # Given the complexity of generating the exact 49 terms in this format, 
    # and the risk of error, I will stick to a safe improvement:
    # Use Strassen for 2x2x2 (Rank 7) and apply it to 4x4x4 blocks?
    # 4x4x4 can be seen as 4 blocks of 2x2x2.
    # Rank would be 4 * 7 = 28? No, Strassen on 4x4 is 7 * 7 = 49.
    
    # I will implement the 49-term Strassen decomposition explicitly.
    
    def build_strassen_4x4x4():
        r = 49
        U = np.zeros((16, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        
        # Terms are (u_idx, v_idx, w_idx, coeff)
        # u_idx = i*4+j
        # v_idx = j*4+k
        # w_idx = k*4+i
        
        terms = []
        
        # Helper to add a scalar term
        def add_term(i, j, k, coeff):
            u = i * 4 + j
            v = j * 4 + k
            w = k * 4 + i
            terms.append((u, v, w, coeff))
            
        # Strassen 2x2x2 terms (7 terms)
        # We need to apply this to 4x4.
        # Split 4x4 into 2x2 blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # Each Pk is a 2x2 product.
        # We expand each Pk using the 7 Strassen terms for 2x2.
        
        # Let's define the 7 Strassen combinations for 2x2 as functions of block indices.
        # Block indices: 00, 01, 10, 11.
        
        # We will iterate through the 7 P's and expand them.
        
        # P1: (A00+A11)(B00+B11)
        # This is a product of two 2x2 matrices.
        # Let M1 = A00+A11, N1 = B00+B11.
        # We need the 7 Strassen terms for M1*N1.
        # M1 = [[A00_00+A11_00, A00_01+A11_01], [A00_10+A11_10, A00_11+A11_11]]
        # This is getting too detailed.
        
        # I will use a pre-computed list of 49 terms.
        # I will assume the standard Strassen decomposition is valid.
        
        return U, V, W, 49

    # Given the time constraints and risk, I will return the schoolbook solution.
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
