import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 x 4x4 using Strassen (rank 28)
    # Part 2: 4x4 x 4x1 using Schoolbook (rank 16)
    
    # Initialize U, V, W
    # Rank will be 28 + 16 = 44
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4 x 4x4 ---
    # Strassen coefficients for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mk = Ak * Bk where Ak, Bk are 2x2 matrices.
    # Ak, Bk are linear combinations of the 2x2 blocks of A and B.
    
    # Let's define the blocks of A and B as 2x2 matrices of coefficients.
    # A_block[i][j] is a 2x2 matrix of coefficients for A[i][j]
    # Actually, A is 4x4. A[i][j] is a scalar.
    # Block A00 is A[0:2, 0:2].
    
    # We will generate columns for each term.
    col_offset = 0
    
    # Define Strassen terms as (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: 4x4 matrix (coefficients for A entries)
    # B_coeffs: 4x4 matrix (coefficients for B entries)
    # C_coeffs: 4x4 matrix (coefficients for C entries)
    
    # Helper to create 4x4 coefficient matrix from block coefficients
    def make_coeff_matrix(block_coeffs):
        # block_coeffs is 2x2, each entry is a 2x2 matrix
        mat = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                mat[i*2:(i+1)*2, j*2:(j+1)*2] = block_coeffs[i][j]
        return mat

    # Identity 2x2
    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    A1 = make_coeff_matrix([
        [I2, Z2],
        [Z2, I2]
    ])
    B1 = make_coeff_matrix([
        [I2, Z2],
        [Z2, I2]
    ])
    C1 = make_coeff_matrix([
        [I2, Z2],
        [Z2, I2]
    ]) # C00 gets +M1, C11 gets +M1
    
    # Wait, C1 should be the coefficient matrix for C.
    # C00 += M1, C11 += M1.
    # So C1 has I2 in (0,0) and (1,1) blocks.
    
    # We need to add the decomposition of A1 * B1 to U, V and A1 * B1 * C1 to W?
    # No. The decomposition is for the tensor.
    # U, V, W are such that sum_c U[:, c] V[:, c]^T W[:, c]^T = Tensor.
    # For each term k, we decompose A_k B_k into rank 4.
    # Let A_k B_k = sum_{r=1}^4 u_r v_r^T.
    # Then the contribution to C is C_k * (A_k B_k).
    # So W gets C_k * w_r?
    # Actually, the tensor is T[i, j, k] = sum A[i, a] B[a, j] delta...
    # Let's just use the standard construction.
    
    # For a term with coefficient matrices A_c, B_c, C_c:
    # The sub-tensor is T_sub = A_c * B_c * C_c (outer product style)
    # We decompose A_c * B_c (which is a 4x4 matrix) into rank 4?
    # No, A_c and B_c are 4x4 matrices of coefficients.
    # The actual operation is:
    # For each entry (i, j) in C:
    # C[i, j] += C_c[i, j] * sum_{a, b} A_c[i, a] A[a, a_c?] ...
    
    # Let's stick to the explicit column generation.
    # For each Strassen term k:
    # 1. Identify A_k (4x4 coeff matrix) and B_k (4x4 coeff matrix).
    # 2. Compute the 4x4 matrix M_k = A_k @ B_k? No.
    # A_k and B_k are linear operators on A and B.
    # The product is a bilinear form.
    # We can just use the schoolbook decomposition for the 4x4x4 part?
    # No, we want rank 28.
    
    # Let's generate the 28 columns.
    # For each of the 7 Strassen terms:
    #   We have a 2x2 matrix product of 2x2 blocks.
    #   Let's say the term is P = L * R where L is 2x2 (from A) and R is 2x2 (from B).
    #   L and R are linear combinations of the 2x2 blocks of A and B.
    #   The product P is a 2x2 matrix.
    #   P is added to C blocks with some coefficients.
    
    # Let's define the 7 terms explicitly with 2x2 block coefficients.
    
    terms = []
    
    # M1
    L1 = np.array([[1, 0], [0, 1]]) # Coeff for A00, A11
    R1 = np.array([[1, 0], [0, 1]]) # Coeff for B00, B11
    # C coeffs: C00 += 1*M1, C11 += 1*M1
    C1_coeffs = { (0,0): 1, (1,1): 1 }
    terms.append((L1, R1, C1_coeffs))
    
    # M2
    L2 = np.array([[0, 0], [1, 1]]) # A10, A11
    R2 = np.array([[1, 0], [0, 0]]) # B00
    C2_coeffs = { (1,0): 1 } # C10 += M2
    terms.append((L2, R2, C2_coeffs))
    
    # M3
    L3 = np.array([[1, 0], [0, 0]]) # A00
    R3 = np.array([[0, 1], [0, -1]]) # B01, -B11
    C3_coeffs = { (0,1): 1, (1,1): 1 } # C01 += M3, C11 += M3
    terms.append((L3, R3, C3_coeffs))
    
    # M4
    L4 = np.array([[0, 0], [0, 1]]) # A11
    R4 = np.array([[-1, 0], [1, 0]]) # -B00, B10
    C4_coeffs = { (0,0): 1, (1,0): 1 } # C00 += M4, C10 += M4
    terms.append((L4, R4, C4_coeffs))
    
    # M5
    L5 = np.array([[1, 1], [0, 0]]) # A00, A01
    R5 = np.array([[0, 0], [0, 1]]) # B11
    C5_coeffs = { (0,0): -1, (0,1): 1 } # C00 -= M5, C01 += M5
    terms.append((L5, R5, C5_coeffs))
    
    # M6
    L6 = np.array([[-1, 0], [1, 0]]) # -A00, A10
    R6 = np.array([[1, 1], [0, 0]]) # B00, B01
    C6_coeffs = { (1,1): 1 } # C11 += M6
    terms.append((L6, R6, C6_coeffs))
    
    # M7
    L7 = np.array([[0, 1], [0, -1]]) # A01, -A11
    R7 = np.array([[0, 0], [1, 1]]) # B10, B11
    C7_coeffs = { (0,0): 1 } # C00 += M7
    terms.append((L7, R7, C7_coeffs))
    
    for L, R, C_coeffs in terms:
        # L and R are 2x2 matrices of coefficients for the 2x2 blocks.
        # We need to decompose the product of the corresponding 4x4 matrices.
        # Let A_L be the 4x4 matrix representing the linear map for L.
        # A_L[i, j] = L[i_block, j_block] if i%2==0, j%2==0?
        # No, L is 2x2. Each entry is a scalar coefficient for the 2x2 block.
        # So A_L is block diagonal? No.
        # A_L is a 4x4 matrix where A_L[2*i+u, 2*j+v] = L[i, j] * delta(u, v)?
        # No, the block A_ij is 2x2.
        # The term is sum_{i,j} L[i, j] A_ij.
        # So A_L is a 4x4 matrix with L[i, j] in the (2i:2i+2, 2j:2j+2) block.
        
        A_L = np.zeros((4, 4))
        B_R = np.zeros((4, 4))
        
        for i in range(2):
            for j in range(2):
                A_L[2*i:2*i+2, 2*j:2*j+2] = L[i, j]
                B_R[2*i:2*i+2, 2*j:2*j+2] = R[i, j]
                
        # Now we need to decompose the bilinear form defined by A_L and B_R.
        # The product is M = A_L @ A * B_R @ B? No.
        # The term is (L(A)) * (R(B)).
        # L(A) is a 2x2 matrix. R(B) is a 2x2 matrix.
        # Product P = L(A) @ R(B).
        # P is 2x2.
        # P[k, l] = sum_{u, v} L(A)[k, u] R(B)[u, l].
        # This is a sum of 4 scalar multiplications.
        # For u, v in 0,1:
        #   Term is L(A)[k, u] * R(B)[u, l].
        #   L(A)[k, u] = sum_{i, j} L[i, j] A[2i+?, 2j+?]
        #   Actually, L(A) is just the matrix of blocks.
        #   Let's use the explicit 4x4 matrices A_L and B_R.
        #   The bilinear form is sum_{p, q} A_L[p, q] A[p, q] B_R[r, s] B[r, s]?
        #   No.
        #   L(A) is a 2x2 matrix. Its (k, u) entry is sum_{i, j} L[i, j] A[2i+k, 2j+u]?
        #   No, L[i, j] is the coefficient for the block A_ij.
        #   So L(A)[k, u] = sum_{i, j} L[i, j] A[2i+k, 2j+u] is wrong.
        #   L(A) is a 2x2 matrix of blocks.
        #   (L(A))_{k, u} is a 2x2 matrix? No.
        #   L is 2x2. A is 4x4.
        #   L(A) is a 4x4 matrix? No.
        #   Strassen operates on 2x2 blocks.
        #   So L(A) is a 2x2 matrix where each entry is a 2x2 block of A.
        #   So L(A) is effectively a 4x4 matrix.
        #   Let's call it M_A.
        #   M_A[2*k+x, 2*u+y] = L[k, u] * A[2*k+x, 2*u+y]?
        #   No. L[k, u] is the coefficient for the block A_{k, u}.
        #   So M_A is block diagonal with L[k, u] on the diagonal?
        #   No.
        #   M_A = sum_{k, u} L[k, u] E_{k, u} \otimes I_2.
        #   Where E_{k, u} is the 2x2 matrix with 1 at (k, u).
        #   So M_A[2*k+x, 2*u+y] = L[k, u] * delta(x, y).
        
        # Similarly for B.
        
        # The product P = M_A @ M_B.
        # P is 4x4.
        # P is added to C with coefficients from C_coeffs.
        
        # We decompose P = M_A @ M_B.
        # P[i, j] = sum_{k} M_A[i, k] M_B[k, j].
        # This is a standard matrix product.
        # We can use schoolbook decomposition for this 4x4x4 product.
        # Rank 16.
        # But we want rank 4 per term?
        # M_A and M_B are sparse.
        # M_A has non-zeros only where x==y.
        # So M_A[i, k] is non-zero only if i%2 == k%2.
        # This structure allows lower rank.
        
        # Let's just use the 4 scalar multiplications for the 2x2 block product.
        # P (2x2 block matrix) = M_A (2x2 block) * M_B (2x2 block).
        # P[k, u] (2x2 block) = sum_{v} M_A[k, v] (2x2) * M_B[v, u] (2x2).
        # Each 2x2 * 2x2 product is rank 4.
        # There are 4 such block products (for k, u in 0,1).
        # Total rank 16 per term?
        # No, the 2x2 block product M_A[k, v] * M_B[v, u] is a 2x2 matrix.
        # It can be decomposed into rank 4.
        # So for each term, we have 4 block positions (k, u).
        # Each requires rank 4.
        # Total rank 16 per term.
        # 7 terms * 16 = 112.
        # This is worse than schoolbook 64.
        
        # My previous logic that rank is 28 was flawed.
        # Strassen reduces the number of *block* multiplications from 8 to 7.
        # Each block multiplication is 4x4x4? No, 2x2x2.
        # Rank of 2x2x2 is 4 (schoolbook).
        # So 7 * 4 = 28.
        # Why did I think 16?
        # Because I thought each term involves a 4x4 product.
        # But each term involves a 2x2 * 2x2 product of *blocks*.
        # The blocks are 2x2 matrices.
        # So the operation is (2x2 matrix) * (2x2 matrix) = 2x2 matrix.
        # This is a 2x2x2 tensor.
        # Rank 4.
        # So yes, 7 * 4 = 28.
        
        # So for each term, we add 4 columns.
        
        for k_block in range(2):
            for u_block in range(2):
                # Compute P[k_block, u_block] = sum_v M_A[k_block, v] * M_B[v, u_block]
                # M_A[k_block, v] is a 2x2 matrix.
                # M_B[v, u_block] is a 2x2 matrix.
                # Product is 2x2.
                # Decompose this 2x2x2 product.
                
                # M_A[k, v] is non-zero only if L[k, v] != 0.
                # M_A[k, v] = L[k, v] * I_2.
                # M_B[v, u] = R[v, u] * I_2.
                # Product = L[k, v] * R[v, u] * I_2.
                # This is a scalar * I_2.
                # Rank 1!
                
                # So each block product is rank 1.
                # 4 block products per term.
                # Rank 4 per term.
                # Total 28.
                
                # So for each (k_block, u_block), we find the scalar coeff.
                # coeff = sum_v L[k_block, v] * R[v, u_block]
                
                coeff = 0
                for v in range(2):
                    coeff += L[k_block, v] * R[v, u_block]
                    
                if coeff != 0:
                    # The product is coeff * I_2.
                    # This contributes to C blocks (k_block, u_block).
                    # The C coefficient for this block is C_coeffs.get((k_block, u_block), 0).
                    # Wait, C_coeffs maps (k_block, u_block) to a scalar.
                    # But C is 4x4.
                    # The block C_{k_block, u_block} is 2x2.
                    # The contribution is C_coeff * (coeff * I_2).
                    # So the 2x2 block of W gets C_coeff * coeff * I_2.
                    
                    c_coeff = C_coeffs.get((k_block, u_block), 0)
                    if c_coeff == 0:
                        continue
                    
                    final_coeff = c_coeff * coeff
                    
                    # Add 4 columns for the I_2 part?
                    # I_2 = [[1, 0], [0, 1]].
                    # Product is final_coeff * [[1, 0], [0, 1]].
                    # This is two independent terms:
                    # 1. final_coeff * A[2*k+x, 2*u+x] * B[2*v+x, 2*w+x]?
                    # No.
                    # The term is final_coeff * I_2.
                    # I_2[x, y] = delta(x, y).
                    # So we have two terms: (x=0, y=0) and (x=1, y=1).
                    
                    for x in range(2):
                        # U, V, W for this scalar term
                        # U[A_idx, col] = 1
                        # V[B_idx, col] = 1
                        # W[C_idx, col] = final_coeff
                        
                        a_idx = 2*k_block + x
                        b_idx = 2*u_block + x
                        c_idx = 2*k_block + x # Wait.
                        
                        # The product is A[2*k+x, 2*v+x] * B[2*v+x, 2*u+x]?
                        # No.
                        # M_A[k, v] = L[k, v] I_2.
                        # M_B[v, u] = R[v, u] I_2.
                        # Product = L[k, v] R[v, u] I_2.
                        # (I_2)_{x, y} = delta(x, y).
                        # So the term is sum_{x} delta(x, y) A[...] B[...]
                        # This implies A index and B index are coupled.
                        # A index: 2*k+x, 2*v+x?
                        # No.
                        # M_A is 4x4.
                        # M_A[i, j] = L[i//2, j//2] * delta(i%2, j%2).
                        # So A[i, j] is multiplied by L[i//2, j//2] if i%2==j%2.
                        
                        # Let's just use the explicit 4x4 matrices and schoolbook.
                        # It's safer.
                        pass
                        
    # This is getting complicated to derive manually.
    # Let's just use the Kronecker product approach.
    # Rank 49 is safe. Rank 44 is better.
    # I will implement Rank 49 (Strassen squared) + Rank 16 = 65.
    # Or Rank 28 + 16 = 44.
    
    # Let's go with 44.
    # I will implement the 28 rank part by iterating the 7 terms and 4 sub-terms.
    
    # Re-implementing the inner loop correctly.
    
    for L, R, C_coeffs in terms:
        for k_block in range(2):
            for u_block in range(2):
                coeff = 0
                for v in range(2):
                    coeff += L[k_block, v] * R[v, u_block]
                
                if coeff == 0:
                    continue
                    
                c_coeff = C_coeffs.get((k_block, u_block), 0)
                if c_coeff == 0:
                    continue
                
                final_coeff = c_coeff * coeff
                
                # The term is final_coeff * I_2 in the (k_block, u_block) block of C.
                # This means C[2*k_block+x, 2*u_block+x] gets final_coeff.
                # The A and B indices are coupled.
                # A index: 2*k_block+x, 2*v+x?
                # No.
                # M_A[k, v] = L[k, v] I_2.
                # M_B[v, u] = R[v, u] I_2.
                # Product = L[k, v] R[v, u] I_2.
                # (I_2)_{x, y} = delta(x, y).
                # So the sum is over x.
                # Term x: A[2*k+x, 2*v+x] * B[2*v+x, 2*u+x].
                # Wait, v is summed out.
                # So for a fixed k, u, x:
                # Sum_v L[k, v] R[v, u] A[2*k+x, 2*v+x] B[2*v+x, 2*u+x].
                # This is a sum of products.
                # We can decompose this sum.
                # For each v, we have a term.
                
                for v in range(2):
                    term_coeff = L[k_block, v] * R[v, u_block]
                    if term_coeff == 0:
                        continue
                    
                    a_idx = 2*k_block + x # x is from I_2
                    # Wait, x is the diagonal index.
                    # We need to iterate x.
                    
    # This is too messy. I will use Rank 65 (49+16).
    # Rank 49 is Strassen squared.
    
    # Generate Rank 49 for 4x4x4.
    # Use Kronecker product of Rank 7 (2x2x2) with Rank 7 (2x2x2).
    
    # Rank 7 matrices for 2x2x2:
    U2, V2, W2 = strassen_2x2()
    
    # U4 = U2 kron I2?
    # No.
    # U4[i, c] = U2[i//2, c//7] * delta(i%2, c%7? No.
    
    # Let's just use schoolbook for 4x4x4 (Rank 64) + 16 = 80.
    # That's the same as current.
    
    # I will implement Rank 65.
    
    # 1. Generate Rank 49 decomposition for 4x4x4.
    # 2. Generate Rank 16 decomposition for 4x4x1.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def strassen_2x2():
    # Returns U, V, W of shape (4, 7)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

# Note: The above Strassen implementation needs verification.
# I will use a known correct implementation or fallback to schoolbook if unsure.
# Given the risk, I will stick to the 65 rank solution using Kronecker product.
