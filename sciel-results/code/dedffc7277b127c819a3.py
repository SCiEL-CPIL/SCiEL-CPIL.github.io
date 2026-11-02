import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 4x4x4 (rank 49) and 4x4x3 (rank 48)
    # Total rank 97
    
    # Part 1: 4x4x4 using Strassen-Strassen (rank 49)
    # Indices for A: 0..15 (4x4)
    # Indices for B: 0..15 (4x4)
    # Indices for C: 0..15 (4x4)
    
    U_part1 = []
    V_part1 = []
    W_part1 = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U_part1.append(u)
        V_part1.append(v)
        W_part1.append(w)
        
    # Strassen decomposition for 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # A00 is top-left 2x2, etc.
    
    # The 7 block products M1..M7
    # Each Mk is a 2x2 matrix product of linear combinations.
    # We define the linear combinations for A and B for each Mk.
    # Then for each Mk, we apply Strassen to the 2x2 matrices.
    
    # Let's define the 7 block-level Strassen products.
    # Each product is of the form (A_comb) * (B_comb)
    # Where A_comb is a 2x2 matrix formed by adding/subtracting blocks of A.
    # And B_comb is a 2x2 matrix formed by adding/subtracting blocks of B.
    
    # Block indices in 4x4 matrix:
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # We represent a 2x2 block combination as a 4x4 matrix of coefficients for A.
    # Actually, it's easier to think in terms of the 4x4 matrix A.
    
    # Let's define the 7 A-combinations and B-combinations.
    # A_comb1 = A00 + A11
    # A_comb2 = A10 + A11
    # A_comb3 = A00
    # A_comb4 = A11
    # A_comb5 = A00 + A01
    # A_comb6 = A10 - A00
    # A_comb7 = A01 - A11
    
    # B_comb1 = B00 + B11
    # B_comb2 = B00
    # B_comb3 = B01 - B11
    # B_comb4 = B10 - B00
    # B_comb5 = B11
    # B_comb6 = B00 + B01
    # B_comb7 = B10 + B11
    
    # For each k in 1..7, we have A_k and B_k.
    # We compute M_k = A_k * B_k using Strassen.
    # Strassen on 2x2 matrices A_k and B_k produces 7 scalar products.
    # Each scalar product is a term in the tensor decomposition.
    
    # Let's create the 4x4 matrices for A_k and B_k.
    # A_k will be a 4x4 matrix with 0s, 1s, -1s.
    
    A_combinations = []
    B_combinations = []
    
    # A00 is rows 0,1 cols 0,1
    # A01 is rows 0,1 cols 2,3
    # A10 is rows 2,3 cols 0,1
    # A11 is rows 2,3 cols 2,3
    
    def get_block(A_mat, r_start, r_end, c_start, c_end):
        return A_mat[r_start:r_end, c_start:c_end]
        
    # We work with 4x4 matrices.
    # A_comb1 = A00 + A11
    A1 = np.zeros((4,4))
    A1[0:2, 0:2] = 1
    A1[2:4, 2:4] = 1
    A_combinations.append(A1)
    
    # A_comb2 = A10 + A11
    A2 = np.zeros((4,4))
    A2[2:4, 0:2] = 1
    A2[2:4, 2:4] = 1
    A_combinations.append(A2)
    
    # A_comb3 = A00
    A3 = np.zeros((4,4))
    A3[0:2, 0:2] = 1
    A_combinations.append(A3)
    
    # A_comb4 = A11
    A4 = np.zeros((4,4))
    A4[2:4, 2:4] = 1
    A_combinations.append(A4)
    
    # A_comb5 = A00 + A01
    A5 = np.zeros((4,4))
    A5[0:2, 0:4] = 1
    A_combinations.append(A5)
    
    # A_comb6 = A10 - A00
    A6 = np.zeros((4,4))
    A6[2:4, 0:2] = 1
    A6[0:2, 0:2] = -1
    A_combinations.append(A6)
    
    # A_comb7 = A01 - A11
    A7 = np.zeros((4,4))
    A7[0:2, 2:4] = 1
    A7[2:4, 2:4] = -1
    A_combinations.append(A7)
    
    # B_combinations
    # B00 is rows 0,1 cols 0,1
    # B01 is rows 0,1 cols 2,3
    # B10 is rows 2,3 cols 0,1
    # B11 is rows 2,3 cols 2,3
    
    B1 = np.zeros((4,4))
    B1[0:2, 0:2] = 1
    B1[2:4, 2:4] = 1
    B_combinations.append(B1)
    
    B2 = np.zeros((4,4))
    B2[0:2, 0:2] = 1
    B_combinations.append(B2)
    
    B3 = np.zeros((4,4))
    B3[0:2, 2:4] = 1
    B3[2:4, 2:4] = -1
    B_combinations.append(B3)
    
    B4 = np.zeros((4,4))
    B4[2:4, 0:2] = 1
    B4[0:2, 0:2] = -1
    B_combinations.append(B4)
    
    B5 = np.zeros((4,4))
    B5[2:4, 2:4] = 1
    B_combinations.append(B5)
    
    B6 = np.zeros((4,4))
    B6[0:2, 0:2] = 1
    B6[0:2, 2:4] = 1
    B_combinations.append(B6)
    
    B7 = np.zeros((4,4))
    B7[2:4, 0:2] = 1
    B7[2:4, 2:4] = 1
    B_combinations.append(B7)
    
    # Now, for each k in 1..7, we have A_k and B_k.
    # We need to decompose the product A_k * B_k using Strassen.
    # A_k and B_k are 4x4 matrices, but they are block-diagonal or block-structured?
    # No, A_k is a 4x4 matrix.
    # But the product A_k * B_k is a 4x4 matrix product.
    # We can apply Strassen to this 4x4 product?
    # No, Strassen is for 2x2.
    # Wait, A_k is formed from blocks.
    # A_k * B_k is the product of two 4x4 matrices.
    # We want to express this product as a sum of rank-1 tensors.
    # But we already used 7 terms for the block level.
    # The block level Strassen says C = sum of 7 block products.
    # Each block product is a 2x2 matrix product.
    # So A_k and B_k are actually 2x2 matrices in the block sense?
    # No.
    # Let's re-read Strassen.
    # C11 = M1 + M4 - M5 + M7
    # M1 is a 2x2 matrix product.
    # So M1 = A' * B' where A' and B' are 2x2 matrices.
    # A' is formed from A blocks.
    # So A' is a 2x2 matrix whose entries are 2x2 blocks?
    # No, A' is a 2x2 matrix of scalars?
    # No, A' is a 2x2 matrix of blocks.
    # So A' * B' is a 2x2 matrix of blocks.
    # Each block is a 2x2 matrix product.
    # So M1 is a 4x4 matrix.
    # And M1 is computed as A' * B'.
    # A' is a 4x4 matrix (formed by 2x2 blocks).
    # B' is a 4x4 matrix.
    # So M1 is a 4x4 matrix product.
    # To compute M1, we can use schoolbook (16 mults) or Strassen (7 mults of 2x2 blocks? No).
    # Strassen reduces 2x2 matrix multiplication.
    # Here, A' and B' are 2x2 matrices of 2x2 blocks.
    # So we are doing 2x2 matrix multiplication where the "scalars" are 2x2 matrices.
    # So we use 7 "scalar" multiplications, where each "scalar" is a 2x2 matrix.
    # So we need 7 products of 2x2 matrices.
    # Each 2x2 matrix product can be done with Strassen (7 scalar mults).
    # So total 7 * 7 = 49 scalar mults.
    
    # So, for each k in 1..7:
    # We have A_k (4x4) and B_k (4x4).
    # We view A_k as a 2x2 matrix of 2x2 blocks.
    # A_k = [[A_k00, A_k01], [A_k10, A_k11]]
    # We apply Strassen to compute A_k * B_k.
    # This generates 7 terms, each corresponding to a 2x2 matrix product.
    # Each 2x2 matrix product is further decomposed using Strassen into 7 scalar terms.
    # So for each k, we generate 7 * 7 = 49 terms?
    # No, 7 block products, each decomposed into 7 scalar products.
    # So 7 * 7 = 49 terms per k?
    # No.
    # The block-level Strassen has 7 products M1..M7.
    # Each Mj is a 2x2 matrix product of blocks.
    # So Mj = A_j * B_j where A_j, B_j are 2x2 matrices of 2x2 blocks.
    # To compute Mj, we use Strassen on the 2x2 block matrices.
    # This requires 7 block products.
    # Each block product is a 2x2 matrix multiplication.
    # To compute a 2x2 matrix multiplication, we use Strassen again.
    # This requires 7 scalar products.
    # So for each Mj, we have 7 block products, each of rank 7 (scalar).
    # Total rank for Mj is 7 * 7 = 49.
    # And we have 7 Mj's.
    # Total rank 7 * 49 = 343?
    # No.
    # The rank of 4x4x4 is 49.
    # My reasoning is flawed.
    
    # Correct reasoning:
    # Rank of 4x4x4 is 49.
    # This is achieved by a specific algorithm.
    # It is NOT simply Strassen applied twice.
    # Strassen applied twice would be $7^2 = 49$ multiplications.
    # Yes, it is 49.
    # But the structure is:
    # 1. Block-level Strassen: 7 products of 2x2 block matrices.
    # 2. Each 2x2 block matrix product is computed using Strassen.
    # 3. Strassen on 2x2 matrices uses 7 scalar products.
    # So total scalar products = 7 (block products) * 7 (scalar products per block product) = 49.
    
    # So, for each of the 7 block products M1..M7:
    # We need to decompose the 2x2 block matrix product A_j * B_j.
    # A_j and B_j are 2x2 matrices of 2x2 blocks.
    # Let A_j = [[A_j00, A_j01], [A_j10, A_j11]]
    # Let B_j = [[B_j00, B_j01], [B_j10, B_j11]]
    # The Strassen decomposition of A_j * B_j uses 7 products:
    # P1 = (A_j00 + A_j11)(B_j00 + B_j11)
    # ...
    # Each Pk is a product of two 2x2 matrices.
    # Each 2x2 matrix product is decomposed using Strassen into 7 scalar products.
    # So for each Pk, we have 7 scalar terms.
    # Total terms per Mj is 7 * 7 = 49.
    # And we have 7 Mj's.
    # Total terms = 7 * 49 = 343.
    # This is wrong. The rank is 49.
    
    # Where is the error?
    # The rank of 4x4x4 is 49.
    # Strassen's algorithm for 4x4 matrices uses $7^2 = 49$ multiplications.
    # This implies that the tensor decomposition has rank 49.
    # The structure is:
    # The 49 multiplications are the scalar products.
    # So we just need to list the 49 scalar products.
    # They are generated by applying Strassen twice.
    
    # Let's trace one term.
    # Term is defined by $u, v, w$.
    # $u$ is a vector of length 16.
    # $v$ is a vector of length 16.
    # $w$ is a vector of length 16.
    
    # I will implement the generation of these 49 terms.
    
    # Part 2: 4x4x3 using schoolbook (rank 48)
    
    # Total rank 97.
    
    pass
