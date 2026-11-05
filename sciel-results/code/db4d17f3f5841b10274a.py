import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4 * 4x4
    rank1 = 28
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * 4)]
    W1 = [[0] * rank1 for _ in range(4 * n)]
    
    # Part 2: 4x4 * 4x2
    rank2 = 16
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * 2)]
    W2 = [[0] * rank2 for _ in range(2 * n)]
    
    # Part 3: 4x4 * 4x1
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(n * m)]
    V3 = [[0] * rank3 for _ in range(m * 1)]
    W3 = [[0] * rank3 for _ in range(1 * n)]
    
    # Fill U1, V1, W1 for 4x4 * 4x4
    # Strassen blocks
    # A is 4x4, B is 4x4
    # Blocks are 2x2
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    
    # We define the 7 Strassen terms
    # Each term is a product of a 2x2 matrix from A and a 2x2 matrix from B
    # We use schoolbook for each 2x2 product
    
    # Helper to add a 2x2 * 2x2 product to the decomposition
    # a_offset, b_offset are the top-left indices of the 2x2 block in A and B
    # c_offset is the top-left index of the 2x2 block in C
    # sign is the coefficient
    # col_offset is the current column index in the rank dimension
    
    def add_2x2_product(U, V, W, a_offset, b_offset, c_offset, sign, col_offset, n_block=2):
        # n_block=2
        # A block is at a_offset, B block at b_offset
        # C block at c_offset
        # Schoolbook: 4 products
        # A[i,j] * B[j,k] -> C[i,k]
        # Indices in U: a_offset + i*n_block + j
        # Indices in V: b_offset + j*n_block + k
        # Indices in W: c_offset + k*n_block + i
        
        for i in range(n_block):
            for j in range(n_block):
                for k in range(n_block):
                    # This is not schoolbook. Schoolbook is sum over j.
                    # Rank 1 term: A[i,j] * B[j,k] contributes to C[i,k]
                    # So for each j, we have a product?
                    # No, schoolbook for 2x2 is 4 terms:
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # Each term A[i,j] B[j,k] is a rank 1 tensor?
                    # Yes, if we fix i,j,k.
                    # But we need to sum over j.
                    # So for fixed i,k, we sum over j.
                    # This means we need 4 terms for each i,k?
                    # No, the rank 1 tensor is defined by fixing i,j,k.
                    # The decomposition is sum over all i,j,k of A[i,j] B[j,k] E_{i,k}
                    # So there are 2*2*2 = 8 terms?
                    # No, for 2x2 * 2x2, schoolbook rank is 4.
                    # Why?
                    # Because we can group them?
                    # No, the tensor rank of M_2,2,2 is 7.
                    # Schoolbook decomposition has rank 4?
                    # Wait. M_2,2,2 tensor: T_{i,j,k} = delta_{i,k}? No.
                    # T_{i,j,k} is the coefficient of A_{i,l} B_{l,j} in C_{i,j}.
                    # T_{i,j,k} = sum_l A_{i,l} B_{l,j} / A_{i,k} B_{k,j}? No.
                    # The tensor is T_{i,j,k} = 1 if i,j,k form a valid path?
                    # Actually, the multiplication tensor T_{i,j,k} is defined such that
                    # C_{i,j} = sum_k T_{i,j,k} A_{i,k} B_{k,j}.
                    # Here T_{i,j,k} = 1 if k is the contraction index.
                    # So T_{i,j,k} = delta_{k, l}? No.
                    # The tensor has indices i (row of C), j (col of C), k (contract).
                    # T_{i,j,k} = 1 if the term A_{i,k} B_{k,j} is present.
                    # So for each i,j, there is exactly one k? No, sum over k.
                    # So T_{i,j,k} = 1 for all i,j,k? No.
                    # The tensor is T_{i,j,k} = 1 if we are computing C_{i,j} using A_{i,k} B_{k,j}.
                    # So for each i,j, we sum over k.
                    # So the tensor has non-zero entries at (i,j,k) for all i,j,k?
                    # No, the decomposition is sum_r u_{i,r} v_{k,r} w_{j,r}.
                    # Schoolbook: u_{i,r} = delta_{i, i_r}, v_{k,r} = delta_{k, k_r}, w_{j,r} = delta_{j, j_r}.
                    # We need to match sum_k A_{i,k} B_{k,j}.
                    # So we need terms where i=i, k=k, j=j.
                    # So for each i,j,k, we have a term?
                    # That would be 2*2*2 = 8 terms.
                    # But schoolbook rank is 4.
                    # Why?
                    # Because we can combine terms?
                    # No, the rank is the minimum number of terms.
                    # For 2x2, the rank is 7.
                    # So schoolbook (8 terms) is not minimal.
                    # But we are using "schoolbook" as a decomposition, not necessarily minimal.
                    # The rank of the decomposition we construct is the number of columns in U,V,W.
                    # If we use 8 terms for each 2x2 product, the rank is 8.
                    # If we use 4 terms?
                    # Can we decompose 2x2 * 2x2 with rank 4?
                    # No, the rank is 7.
                    # So we must use at least 7 terms for each 2x2 product.
                    # So for 4x4 * 4x4, using 7 products of 2x2, each of rank 7, total rank 49.
                    # My previous calculation of 28 was based on rank 4 for 2x2.
                    # But rank of 2x2 is 7.
                    # So rank 49 is correct for Strassen-Schoolbook hybrid?
                    # Wait, if I use Strassen for the outer 4x4 (7 terms) and Schoolbook for inner 2x2 (8 terms), total 56.
                    # If I use Strassen for outer and Strassen for inner, total 49.
                    # So rank 49 is achievable.
                    
                    # Let's re-evaluate the rank of M_2,2,2.
                    # It is 7.
                    # So we should use the Strassen decomposition for the inner 2x2 products too.
                    # So for Part 1 (4x4 * 4x4):
                    # 7 outer terms, each is 2x2 * 2x2.
                    # Each inner 2x2 * 2x2 has rank 7.
                    # Total rank 7 * 7 = 49.
                    
                    # For Part 2 (4x4 * 4x2):
                    # Split into 4 blocks of 2x2 * 2x2?
                    # No, B is 4x2.
                    # B11 (2x2), B21 (2x2).
                    # C11 = A11 B11 + A12 B21.
                    # C21 = A21 B11 + A22 B21.
                    # Each term is 2x2 * 2x2.
                    # 4 terms.
                    # Each rank 7.
                    # Total rank 28.
                    
                    # For Part 3 (4x4 * 4x1):
                    # B is 4x1.
                    # B11 (2x1), B21 (2x1).
                    # C11 = A11 B11 + A12 B21.
                    # C21 = A21 B11 + A22 B21.
                    # Each term is 2x2 * 2x1.
                    # Rank of M_2,2,1?
                    # Schoolbook: 4 terms.
                    # Is rank < 4?
                    # M_2,2,1 tensor: T_{i,j,k} = 1 if i,j,k valid.
                    # C_{i,j} = sum_k A_{i,k} B_{k,j}.
                    # B is 2x1. So j=0.
                    # C_{i,0} = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}.
                    # This is 2 terms for each i.
                    # Total 4 terms.
                    # Rank is 4.
                    # So 4 terms * 4 blocks = 16.
                    
                    # Total rank: 49 + 28 + 16 = 93.
                    # Score 112 / 93 = 1.20.
                    
                    # Can we do better?
                    # For Part 1, can we use a better decomposition for 4x4 * 4x4?
                    # Rank is known to be 16? No, 16 is schoolbook.
                    # Rank 49 is better.
                    
                    # Let's implement the rank 49, 28, 16 decomposition.
                    pass
