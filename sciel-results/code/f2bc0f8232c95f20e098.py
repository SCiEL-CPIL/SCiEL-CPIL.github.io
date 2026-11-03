import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 98
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We generate the 49 terms for 4x4x4
    # And duplicate for the second half of B
    
    # Helper to get Strassen 2x2 coefficients
    # Returns list of 7 tuples: (coeffs_x, coeffs_y, coeffs_z)
    # coeffs_x: 4 floats for x00, x01, x10, x11
    # coeffs_y: 4 floats for y00, y01, y10, y11
    # coeffs_z: 4 floats for z00, z01, z10, z11 (reconstruction)
    def strassen_2x2():
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # Reconstruction:
        # Z00 = P1 - P4 + P5 - P7
        # Z01 = P3 + P7
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        terms = []
        
        # P1
        cx = [1, 0, 0, 1]
        cy = [1, 0, 0, 1]
        cz = [1, 0, 0, 1] # P1 contributes to Z00 and Z11
        # Wait, cz is for the final Z.
        # P1 is used in Z00 and Z11.
        # So we need to add P1 to Z00 and Z11.
        
        # It's better to return the linear forms for P and the mapping to Z.
        # Let's return list of (cx, cy, cz_vector) where cz_vector indicates which Z components get this P.
        
        # P1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])) # Z00, Z11
        # P2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1])) # Z10, Z11
        # P3
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1])) # Z01, Z11
        # P4
        terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0])) # Z00, Z10
        # P5
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0])) # Z00
        # P6
        terms.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1])) # Z11
        # P7
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, -1])) # Z01, Z11
        
        return terms

    # Top level 4x4 Strassen
    # X_k and Y_k are 2x2 matrices of blocks.
    # We need to map block indices to scalar indices.
    # Block (b_i, b_j) contains A[i*2 + u, j*2 + v]
    
    # Define the 7 top-level terms
    # Each term is (cx_blocks, cy_blocks)
    # cx_blocks is 4x4 matrix? No, 4 blocks, each 2x2.
    # Let's represent cx as a dict or flat list of 16 coeffs?
    # Actually, X_k is a 2x2 matrix of blocks.
    # X_k[u,v] is a 2x2 block of A.
    # So X_k has 4 blocks. Each block is 2x2. Total 16 entries.
    
    # Let's define the top-level linear forms on the blocks.
    # X1 = A00 + A11
    # X2 = A10 + A11
    # X3 = A00
    # X4 = A11
    # X5 = A00 + A01
    # X6 = A10 - A00
    # X7 = A01 - A11
    
    # Similarly for Y.
    
    # We will iterate through the 7 top-level terms.
    # For each, we get the 2x2 matrix of blocks for X and Y.
    # Then we apply the bottom-level Strassen to these 2x2 matrices of blocks.
    # But the "entries" of the 2x2 matrix of blocks are themselves 2x2 matrices of scalars.
    # So the bottom-level Strassen operates on these 2x2 matrices.
    # The "scalar" multiplications in the bottom level are actually 2x2 matrix multiplications?
    # No.
    
    # Let's be precise.
    # Top level:
    # M_k = X_k Y_k.
    # X_k is a 2x2 matrix where each entry is a 2x2 matrix (block of A).
    # So X_k is effectively a 4x4 matrix? No.
    # X_k is a linear combination of A's blocks.
    # Example: X1 = A00 + A11.
    # A00 is A[0:2, 0:2]. A11 is A[2:4, 2:4].
    # So X1 is a 4x4 matrix? No, X1 is a 2x2 matrix of blocks.
    # When we say "X_k Y_k", we mean block multiplication.
    # (X_k Y_k)_{uv} = X_k[u,0] Y_k[0,v] + X_k[u,1] Y_k[1,v].
    # Each term is a 2x2 matrix product.
    
    # So, for each top-level term k:
    # We have 4 block products to compute?
    # No, Strassen computes the 7 products P_s which are 2x2 matrix products.
    # Then Z (the result 2x2 matrix of blocks) is a linear combination of P_s.
    
    # So, for each k:
    # 1. Determine the 2x2 matrix of blocks X_k and Y_k.
    # 2. Apply Strassen 2x2 logic to these block matrices.
    #    This gives 7 "block products" P_s = L_x(X_k) L_y(Y_k).
    #    Here L_x is a linear combination of the 4 blocks of X_k.
    #    So L_x is a 2x2 matrix of blocks (linear combination of A's blocks).
    #    Similarly L_y.
    #    So P_s is a product of two 2x2 matrices of blocks.
    #    Wait. If L_x is a 2x2 matrix of blocks, and L_y is a 2x2 matrix of blocks,
    #    their product is a 2x2 matrix of blocks.
    #    Computing this product requires another level of Strassen (7 scalar multiplications of blocks? No, blocks are 2x2 matrices).
    #    So we need to decompose the product of two 2x2 matrices (which are 2x2 blocks) into scalar multiplications.
    #    This is exactly the bottom-level Strassen.
    
    # So the structure is:
    # 1. Top-level Strassen (7 terms).
    # 2. For each top-level term, we have a product of two 2x2 matrices (of blocks).
    # 3. We decompose this product using bottom-level Strassen (7 terms).
    # 4. Each bottom-level term is a product of two linear forms of the blocks.
    #    A linear form of blocks is a 2x2 matrix of blocks? No.
    #    In Strassen, the linear forms are on the *entries* of the 2x2 matrix.
    #    Here the "entries" are 2x2 blocks.
    #    So a linear form is a linear combination of the 4 blocks.
    #    The result is a single block (2x2 matrix).
    #    So the bottom-level term is a product of two blocks (2x2 matrices).
    #    Wait, no.
    
    # Let's re-read Strassen.
    # $P_1 = (X_{00}+X_{11})(Y_{00}+Y_{11})$.
    # $X_{00}$ is an entry.
    # If the entries are blocks, then $P_1$ is a product of two blocks.
    # A product of two 2x2 matrices requires 4 scalar multiplications (schoolbook) or 7 (Strassen).
    # Since we want rank 49, we must use Strassen for the block product too.
    # So, for each of the 7 top-level terms, we compute a product of two 2x2 matrices (blocks).
    # We use Strassen to compute this block product, which takes 7 scalar multiplications.
    # Total 7 * 7 = 49.
    
    # So, for each top-level term k:
    # 1. Get the 2x2 block matrices X_k and Y_k.
    # 2. Apply Strassen to X_k and Y_k.
    #    This generates 7 scalar multiplications.
    #    Each scalar multiplication is between a linear form of the blocks of X_k and a linear form of the blocks of Y_k.
    #    Let $L_x$ be the linear form on X_k's blocks.
    #    $L_x = \sum_{u,v} c_{uv} X_k[u,v]$.
    #    $X_k[u,v]$ is a 2x2 block of A.
    #    So $L_x$ is a 2x2 matrix (linear combination of A's blocks).
    #    Similarly $L_y$ is a 2x2 matrix (linear combination of B's blocks).
    #    The scalar multiplication is $L_x \cdot L_y$?
    #    No, Strassen computes $P = L_x L_y$.
    #    $P$ is a 2x2 matrix.
    #    The scalar multiplications are inside the computation of $P$.
    #    Wait, Strassen for 2x2 matrices $L_x, L_y$ computes $P = L_x L_y$ using 7 scalar multiplications of the *entries* of $L_x$ and $L_y$.
    #    The entries of $L_x$ are scalars (linear combinations of A's entries).
    #    So yes, the 7 scalar multiplications are between linear forms of A's entries and linear forms of B's entries.
    
    # So the algorithm is:
    # 1. For each top-level term k (0..6):
    #    a. Determine X_k (2x2 matrix of blocks) and Y_k (2x2 matrix of blocks).
    #    b. Apply Strassen 2x2 to X_k and Y_k.
    #       This gives 7 pairs of linear forms on the *blocks*.
    #       Let's call them $Lx_s$ and $Ly_s$.
    #       $Lx_s$ is a linear combination of the 4 blocks of X_k.
    #       $Ly_s$ is a linear combination of the 4 blocks of Y_k.
    #    c. For each $s$ (0..6):
    #       $Lx_s$ is a 2x2 matrix of A's entries.
    #       $Ly_s$ is a 2x2 matrix of B's entries.
    #       We need to compute the product $Lx_s Ly_s$?
    #       No, Strassen's $P_s$ IS the product.
    #       But we need the scalar multiplications that compute $P_s$.
    #       Wait, Strassen computes $P_s$ as a linear combination of 7 scalar products.
    #       So we need to apply Strassen *again* to the entries of $Lx_s$ and $Ly_s$?
    #       No. $Lx_s$ is a 2x2 matrix. $Ly_s$ is a 2x2 matrix.
    #       The product $Lx_s Ly_s$ is a 2x2 matrix.
    #       To compute this product, we use Strassen (7 scalar multiplications).
    #       So, for each $s$, we perform a Strassen multiplication of the 2x2 matrices $Lx_s$ and $Ly_s$.
    #       This generates 7 scalar multiplications.
    #       Total 7 * 7 = 49.
    
    # This confirms the structure.
    
    # Implementation:
    # 1. Define top-level block linear forms.
    # 2. Define bottom-level Strassen scalar linear forms.
    # 3. Compose them.

    # Top-level block forms (coeffs for A00, A01, A10, A11)
    # A00: A[0:2, 0:2], A01: A[0:2, 2:4], A10: A[2:4, 0:2], A11: A[2:4, 2:4]
    
    # X1 = A00 + A11 -> [1, 0, 0, 1]
    # X2 = A10 + A11 -> [0, 0, 1, 1]
    # X3 = A00       -> [1, 0, 0, 0]
    # X4 = A11       -> [0, 0, 0, 1]
    # X5 = A00 + A01 -> [1, 1, 0, 0]
    # X6 = A10 - A00 -> [-1, 0, 1, 0]
    # X7 = A01 - A11 -> [0, 1, 0, -1]
    
    top_x = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    top_y = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Bottom-level Strassen linear forms (coeffs for x00, x01, x10, x11)
    bot_x = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    bot_y = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Reconstruction coeffs for bottom level (for Z00, Z01, Z10, Z11)
    bot_z = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, -1]
    ]
    
    # Reconstruction coeffs for top level
    top_z = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, -1]
    ]
    
    # Helper to map block index and entry index to global A index
    def a_index(block, u, v):
        # block 0: 00, 1: 01, 2: 10, 3: 11
        bi = block // 2
        bj = block % 2
        i = bi * 2 + u
        j = bj * 2 + v
        return i * 4 + j
        
    def b_index(block, u, v):
        bi = block // 2
        bj = block % 2
        i = bi * 2 + u
        j = bj * 2 + v
        return i * 8 + j # B is 4x8, but we are in 4x4 half, so j is 0..3
        
    # We will fill U, V, W
    # U shape (16, 98)
    # V shape (32, 98)
    # W shape (32, 98)
    
    col = 0
    for k in range(7):
        # Top level term k
        # X_k is 2x2 matrix of blocks.
        # X_k[u,v] = sum_{b} top_x[k][b] * A_block[b]
        # We need to apply bottom-level Strassen to X_k and Y_k.
        # The "entries" of X_k are the blocks.
        # So bot_x[s] applies to the 4 blocks of X_k.
        
        for s in range(7):
            # Bottom level term s
            # Lx = sum_{b} bot_x[s][b] * X_k[b]
            # X_k[b] is a 2x2 block of A.
            # X_k[b] = sum_{b2} top_x[k][b2] * A_block[b2]
            # So Lx = sum_{b, b2} bot_x[s][b] top_x[k][b2] * A_block[b2]
            # Wait, X_k is a 2x2 matrix of blocks.
            # bot_x[s] is a linear form on the 4 entries of X_k (which are blocks).
            # So Lx is a linear combination of the 4 blocks of X_k.
            # Result Lx is a 2x2 matrix (linear combination of A's blocks).
            
            # We need the coefficients of A's entries in Lx.
            # Lx[u,v] (entry u,v of the 2x2 result)
            # Lx = sum_{b} bot_x[s][b] * X_k[b]
            # X_k[b] is a block (2x2).
            # So Lx[u,v] = sum_{b} bot_x[s][b] * X_k[b][u,v]
            # X_k[b][u,v] = sum_{b2} top_x[k][b2] * A_block[b2][u,v]
            
            # So coeff of A_block[b2][u,v] in Lx[u,v] is:
            # sum_{b} bot_x[s][b] * top_x[k][b2]
            
            # Similarly for Ly.
            
            # Now we have Lx (2x2 matrix of linear forms of A) and Ly (2x2 matrix of linear forms of B).
            # We need to compute the product Lx * Ly using Strassen (7 scalar multiplications).
            # This is the same as applying Strassen to the 2x2 matrices Lx and Ly.
            
            # So we iterate t from 0 to 6.
            for t in range(7):
                # Scalar multiplication t
                # Linear form on Lx entries: bot_x[t]
                # Linear form on Ly entries: bot_y[t]
                
                # Lx_lin = sum_{u,v} bot_x[t][u*2+v] * Lx[u,v]
                # Lx[u,v] is a linear form of A.
                # So Lx_lin is a linear form of A.
                
                # Coeffs of A in Lx_lin:
                # a_coeffs = [0]*16
                for i in range(4):
                    for j in range(4):
                        # A[i,j]
                        # Which block? b2 = (i//2)*2 + (j//2)
                        # Entry in block: u = i%2, v = j%2
                        b2 = (i//2)*2 + (j//2)
                        u = i % 2
                        v = j % 2
                        
                        # Coeff of A_block[b2][u,v] in Lx[u,v]
                        # is sum_{b} bot_x[s][b] * top_x[k][b2]
                        coeff_in_Lx = 0
                        for b in range(4):
                            coeff_in_Lx += bot_x[s][b] * top_x[k][b2]
                        
                        # Coeff of Lx[u,v] in Lx_lin
                        # bot_x[t] is indexed by u,v
                        # Index in bot_x[t] is u*2+v
                        coeff_in_lin = bot_x[t][u*2+v]
                        
                        a_coeffs[i*4+j] = coeff_in_Lx * coeff_in_lin
                        
                # Similarly for B
                # b_coeffs = [0]*32 (but only first 16 for this half)
                b_coeffs = [0]*32
                for i in range(4):
                    for j in range(4):
                        b2 = (i//2)*2 + (j//2)
                        u = i % 2
                        v = j % 2
                        
                        coeff_in_Ly = 0
                        for b in range(4):
                            coeff_in_Ly += bot_y[s][b] * top_y[k][b2]
                        
                        coeff_in_lin = bot_y[t][u*2+v]
                        
                        b_coeffs[i*8+j] = coeff_in_Ly * coeff_in_lin
                        
                # Now we have a_coeffs (16) and b_coeffs (32).
                # The scalar product is sum a_coeffs[i,j] A[i,j] * sum b_coeffs[j,k] B[j,k]
                
                # Fill U, V, W
                for i in range(4):
                    for j in range(4):
                        U[i*4+j, col] = a_coeffs[i*4+j]
                        
                for j in range(4):
                    for k in range(4):
                        V[j*8+k, col] = b_coeffs[j*8+k]
                        
                # W reconstruction
                # The result of this scalar product contributes to Z.
                # Z is the result of Lx * Ly.
                # Z is a 2x2 matrix of blocks.
                # Z is part of the top-level Strassen result.
                
                # The contribution to C[i,k] is:
                # sum_{j} a_coeffs[i,j] b_coeffs[j,k]
                # This is the (i,k) entry of the product of the linear forms.
                
                # We need to add this to W.
                # W[k*n+i, col] should be the coefficient.
                
                # The reconstruction involves top_z and bot_z.
                # The scalar product P_{s,t} contributes to Lx*Ly.
                # Lx*Ly contributes to the top-level term k's result M_k.
                # M_k contributes to C.
                
                # Let's trace the reconstruction.
                # Bottom level: P_{s,t} is a scalar.
                # It contributes to the entries of Z = Lx*Ly.
                # Z[u,v] = sum_{t} bot_z[t][u*2+v] * P_{s,t}
                
                # Top level: M_k = X_k Y_k.
                # M_k is a 2x2 matrix of blocks.
                # M_k[u,v] = sum_{s} top_z[k][u*2+v] * (Lx_s * Ly_s)
                # (Lx_s * Ly_s) is the 2x2 matrix Z computed above.
                # So M_k[u,v] = sum_{s} top_z[k][u*2+v] * Z_s[u,v]
                # Z_s[u,v] = sum_{t} bot_z[t][u*2+v] * P_{s,t}
                
                # So M_k[u,v] = sum_{s,t} top_z[k][u*2+v] bot_z[t][u*2+v] P_{s,t}
                
                # C is a 4x4 matrix.
                # C is composed of 4 blocks: C00, C01, C10, C11.
                # C_{block_i, block_j} = M_k? No.
                # The top-level Strassen computes C = X Y.
                # C is a 2x2 matrix of blocks.
                # C[u,v] = sum_{k} top_z[k][u*2+v] * M_k[u,v]
                
                # So C[u,v] (block) = sum_{k,s,t} top_z[k][u*2+v] top_z[k][u*2+v] bot_z[t][u*2+v] P_{s,t}
                # Wait, top_z is used for both levels?
                # Yes, the Strassen structure is the same.
                
                # So the coefficient of P_{s,t} in C[u,v] (block) is:
                # top_z[k][u*2+v] * top_z[k][u*2+v] * bot_z[t][u*2+v]
                
                # C[u,v] is a 2x2 block of C.
                # C[i,j] where i = u*2+i', j = v*2+j'.
                # The entry C[i,j] gets contribution from C[u,v][i',j'].
                # The block C[u,v] is a 2x2 matrix.
                # The scalar P_{s,t} is a scalar.
                # So it adds to all 4 entries of the block C[u,v].
                
                # So for each u,v in 0..1:
                #   coeff = top_z[k][u*2+v] * bot_z[t][u*2+v]
                #   (Note: top_z is used for the top level reconstruction of M_k into C?
                #    No, top_z[k] is the reconstruction of M_k into C.
                #    So C = sum_k top_z[k] M_k.
                #    And M_k = sum_s top_z[k] (Lx_s Ly_s)? No.
                #    M_k is the k-th term.
                #    C = sum_k top_z[k] M_k.
                #    M_k = Lx_k Ly_k.
                #    Lx_k Ly_k = sum_s top_z[k] (Lx_{k,s} Ly_{k,s})? No.
                #    The bottom level Strassen computes Lx_k Ly_k.
                #    The result is a 2x2 matrix.
                #    The reconstruction of the bottom level product is bot_z.
                #    So (Lx_k Ly_k)[u,v] = sum_t bot_z[t][u*2+v] P_{k,s,t}
                #    Wait, s is the bottom-level term index.
                #    So M_k[u,v] = sum_s bot_z[s][u*2+v] P_{k,s}
                #    And C[u,v] = sum_k top_z[k][u*2+v] M_k[u,v]
                #    So C[u,v] = sum_{k,s} top_z[k][u*2+v] bot_z[s][u*2+v] P_{k,s}
                
                # So the coefficient for P_{k,s} in C[u,v] is top_z[k][u*2+v] * bot_z[s][u*2+v].
                
                # We are in loop k, s, t.
                # P_{k,s,t} is the scalar product.
                # It contributes to M_k[u,v] with coeff bot_z[t][u*2+v].
                # Then M_k[u,v] contributes to C[u,v] with coeff top_z[k][u*2+v].
                # So total coeff for C[u,v] is top_z[k][u*2+v] * bot_z[t][u*2+v].
                
                for u in range(2):
                    for v in range(2):
                        coeff_w = top_z[k][u*2+v] * bot_z[t][u*2+v]
                        if coeff_w != 0:
                            # C[u,v] is a 2x2 block.
                            # Indices i = u*2+i', j = v*2+j'
                            for i_prime in range(2):
                                for j_prime in range(2):
                                    i = u*2 + i_prime
                                    j = v*2 + j_prime
                                    # W index: k*n + i
                                    # Here j is the column of C, which is k in the tensor W[k*n+i]
                                    # Wait, W is indexed by (k, i) where k is B's col, i is A's row.
                                    # C[i,j] = sum_k A[i,k] B[k,j].
                                    # So W[j*n + i, col] = coeff.
                                    W[j*4 + i, col] += coeff_w
                                    
                col += 1
                
    # Duplicate for second half of B
    # U is the same.
    # V: shift indices by 4.
    # W: same.
    
    for c in range(49):
        for i in range(16):
            U[i, c+49] = U[i, c]
        for j in range(4):
            for k in range(4):
                V[j*8 + k + 4, c+49] = V[j*8 + k, c]
        for j in range(4):
            for i in range(4):
                W[j*4 + i, c+49] = W[j*4 + i, c]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
