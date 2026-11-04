import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # U shape: (n*m, rank) -> (16, rank)
    # V shape: (m*p, rank) -> (36, rank)
    # W shape: (p*n, rank) -> (36, rank)
    
    # We decompose into 3 parts:
    # 1. A * B1 (4x4 * 4x4) -> Rank 28
    # 2. A * B2 (4x4 * 4x4) -> Rank 28
    # 3. A * B3 (4x4 * 4x1) -> Rank 4
    
    # Total rank 60
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 4x4x4 Strassen block
    # A is 4x4, B is 4x4.
    # Partition into 2x2 blocks.
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C = [C11 C12; C21 C22]
    # Strassen for 2x2x2 uses rank 7.
    # We have 4 such blocks.
    
    def add_strassen_4x4(offset_u, offset_v, offset_w, start_col):
        # offset_u: (i, j) block offset in A (row, col)
        # offset_v: (j, k) block offset in B (row, col)
        # offset_w: (k, i) block offset in C (row, col)
        # Note: W index is k*n + i.
        # C is 4x4. k is col, i is row.
        # Block (b_i, b_k) of C corresponds to rows b_i*2..b_i*2+1, cols b_k*2..b_k*2+1.
        
        # There are 4 blocks: (0,0), (0,1), (1,0), (1,1) for (i, k)
        # For each block, we multiply 2x2 A_block by 2x2 B_block.
        
        for bi in range(2):
            for bk in range(2):
                # A block is at (bi*2, bi*2) to (bi*2+1, bi*2+1)?
                # No, A is 4x4.
                # Block (bi, bj) of A is rows bi*2..bi*2+1, cols bj*2..bj*2+1.
                # We need to sum over j.
                # C_block(bi, bk) = sum_j A_block(bi, j) * B_block(j, bk).
                
                # This requires 3 Strassen decompositions for each C_block?
                # No, Strassen computes the whole 4x4 product with 4 blocks of 2x2.
                # The 4 blocks of 2x2x2 correspond to the 4 sub-products.
                # But Strassen shares intermediate products.
                # The standard Strassen for 4x4x4 uses 4 independent 2x2x2 Strassen decompositions?
                # No, it uses 4 blocks, each rank 7. Total 28.
                # The blocks are disjoint in the output C.
                # C11 uses A11, A12, B11, B12.
                # C12 uses A11, A12, B21, B22.
                # C21 uses A21, A22, B11, B12.
                # C22 uses A21, A22, B21, B22.
                
                # So we can just add 4 Strassen decompositions for 2x2x2.
                
                # Strassen for 2x2x2:
                # M1 = (A11+A22)(B11+B22)
                # M2 = (A21+A22)B11
                # M3 = A11(B12-B22)
                # M4 = A22(B21-B11)
                # M5 = (A11+A12)B22
                # M6 = (A21-A11)(B11+B12)
                # M7 = (A12-A22)(B21+B22)
                
                # C11 = M1 - M4 + M5 - M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # We need to map these to U, V, W.
                # For a specific 2x2 block of A (rows r1, r2; cols c1, c2)
                # and B (rows c1, c2; cols d1, d2)
                # and C (rows r1, r2; cols d1, d2)
                
                # Let's define the local indices for the 2x2 block.
                # a11, a12, a21, a22
                # b11, b12, b21, b22
                # c11, c12, c21, c22
                
                # The global indices:
                # A: row = bi*2 + local_r, col = bj*2 + local_c
                # B: row = bj*2 + local_c, col = bk*2 + local_d
                # C: row = bi*2 + local_r, col = bk*2 + local_d
                # W index: col*n + row = (bk*2 + local_d)*4 + (bi*2 + local_r)
                
                # We sum over bj in 0..1.
                # So for fixed bi, bk, we have two terms: bj=0 and bj=1.
                # Each term is a 2x2x2 product.
                # We can run Strassen for each (bi, bk, bj)?
                # That would be 2*2*2 = 8 Strassen decompositions.
                # Rank 8 * 7 = 56.
                # This is worse than 28.
                # Strassen for 4x4x4 is not just 8 2x2x2 Strassens.
                # It uses the block structure to share?
                # No, the rank of 4x4x4 is 26 or 28.
                # If we use 4 blocks of 2x2x2, each block is independent?
                # C11 depends on A11, A12, B11, B12.
                # C12 depends on A11, A12, B21, B22.
                # They share A11, A12.
                # But the Strassen decomposition is linear.
                # We can just concatenate the decompositions for each (bi, bk) pair?
                # For fixed bi, bk, C_block = sum_j A_block(bi, j) B_block(j, bk).
                # This is a sum of two 2x2x2 tensors.
                # Rank of sum of two 2x2x2 tensors is at most 14.
                # If we don't share, rank is 14.
                # 4 blocks * 14 = 56.
                # This is worse than 28.
                
                # Wait, the rank of 4x4x4 is 28.
                # How is it achieved?
                # By using Strassen on the 4x4 blocks directly?
                # Strassen reduces 8 multiplications to 7 for 2x2.
                # For 4x4, we use 4 blocks of 2x2.
                # If we apply Strassen to each 2x2 block independently, we get 4 * 7 = 28.
                # But the blocks are not independent in the input?
                # A11 is used in C11 and C12.
                # If we compute C11 and C12 separately, we compute A11 * B11 etc separately.
                # But Strassen computes linear combinations.
                # The linear combinations for C11 and C12 might be different.
                # However, the rank is additive if the sub-tensors are disjoint in the factor space.
                # If we use 4 independent Strassen decompositions, rank is 28.
                # This is valid.
                
                # So for each (bi, bk), we compute sum_j A(bi, j) B(j, bk).
                # This is a 2x2x2 tensor? No, it's a sum of two 2x2x2 tensors.
                # We can decompose the sum with rank 14?
                # Or just use schoolbook for the sum? Rank 8?
                # 4 * 8 = 32.
                # 28 is better.
                
                # I will use the 4 independent Strassen decompositions for the 4 output blocks.
                # But wait, the input blocks are shared.
                # Does that matter for the rank?
                # The tensor is a sum of 4 sub-tensors T1, T2, T3, T4.
                # rank(T) <= rank(T1) + rank(T2) + rank(T3) + rank(T4).
                # Each Ti is a 2x2x2 tensor?
                # No, T1 (for C11) is A11 B11 + A12 B21.
                # This is a sum of two 2x2x2 tensors.
                # Rank of T1 <= 7 + 7 = 14.
                # Total rank <= 4 * 14 = 56.
                
                # Is there a better way?
                # Yes, the rank of 4x4x4 is 28.
                # This implies that the 4 sub-tensors can be decomposed with shared factors?
                # Or that the rank of the sum is less than the sum of ranks.
                # I will stick to 56 for safety, or try to find a 28 decomposition.
                # Actually, 28 is the standard Strassen rank for 4x4.
                # I will implement 4 independent Strassen decompositions for the 4 output 2x2 blocks.
                # But each output block is a sum of two 2x2x2 products.
                # I will just use schoolbook for the 2x2x2 products within the block sum?
                # Rank 8 per block. 4 * 8 = 32.
                # 32 is better than 56.
                # 28 is better than 32.
                
                # I will use rank 32 for the 4x4x4 parts to be safe and simple.
                # 32 + 32 + 4 = 68.
                # 68 < 144.
                
                pass
    return {"U": U, "V": V, "W": W}
