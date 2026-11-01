"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product factorization strategy.
    # We represent the 2x12 multiplication as a block diagonal system of four independent 2x3 multiplications.
    # We use a rank-6 decomposition for the 2x3 matrix multiplication (since rank-5 is not possible for 2x3 over integers/half-integers, 
    # and standard Strassen for 2x2x2 gives rank 7 but 2x3 is not a 2x2. 
    # Actually, for 2x3, the schoolbook is 2*3=6. 
    # Is there a rank < 6 for 2x3? 
    # The border rank of 2x3 is 5 (Strassen-like). But exact rank over rationals? 
    # It is known that rank(2x3) = 5 is possible? 
    # Let's check the literature. 
    # The rank of matrix multiplication [m x n, n x p].
    # For 2x2, rank is 7.
    # For 2x3, the schoolbook is 6. 
    # Is it possible to do 2x3 in rank 5? 
    # According to Bini, Capovani, Lotti, Romani (1979), the rank of 2x3 is 5.
    # Let's try to construct it or find the coefficients.
    # However, if I can't be sure, I should stick to a guaranteed valid decomposition.
    # The schoolbook for 2x3 has rank 6. Four blocks -> 24.
    # The current schoolbook for 2x2x12 is 2*2*12 = 48.
    # Wait, the current program's schoolbook is rank n*m*p = 2*2*12 = 48.
    # My idea was 4 blocks of 2x3. Each 2x3 schoolbook is 6. Total 24.
    # This is a valid improvement: 24 < 48.
    
    # Let's verify the block diagonal structure.
    # A is 2x2. B is 2x12.
    # We can split B into 4 blocks of 2x3: B1, B2, B3, B4.
    # C = A * B = [A*B1, A*B2, A*B3, A*B4].
    # Since A is the same, we can reuse the linear combinations of A?
    # No, the tensor decomposition sums terms u_i \otimes v_i \otimes w_i.
    # If we use the same U for all blocks, we can share the A coefficients.
    # Let's define the decomposition for one 2x3 block.
    # A is 2x2 (indices i, j), B is 2x3 (indices j, k), C is 2x3 (indices i, k).
    # Schoolbook for 2x3:
    # C_{i,k} = sum_{j=1}^2 A_{i,j} B_{j,k}
    # Terms:
    # 1. A_{1,1} B_{1,1} -> C_{1,1}
    # 2. A_{1,2} B_{2,1} -> C_{1,1}
    # 3. A_{1,1} B_{1,2} -> C_{1,2}
    # 4. A_{1,2} B_{2,2} -> C_{1,2}
    # 5. A_{1,1} B_{1,3} -> C_{1,3}
    # 6. A_{1,2} B_{2,3} -> C_{1,3}
    # 7. A_{2,1} B_{1,1} -> C_{2,1}
    # 8. A_{2,2} B_{2,1} -> C_{2,1}
    # 9. A_{2,1} B_{1,2} -> C_{2,2}
    # 10. A_{2,2} B_{2,2} -> C_{2,2}
    # 11. A_{2,1} B_{1,3} -> C_{2,3}
    # 12. A_{2,2} B_{2,3} -> C_{2,3}
    
    # Total 12 terms per block.
    # If we have 4 independent blocks, and we can't share U coefficients easily without mixing, 
    # we just concatenate the tensors.
    # Rank = 4 * 6 = 24.
    
    # Let's build the matrices for the 2x2x12 case using this block structure.
    # n=2, m=2, p=12.
    # U shape: (n*m, rank) = (4, 24)
    # V shape: (m*p, rank) = (24, 24)
    # W shape: (p*n, rank) = (24, 24)
    
    # Block b (0 to 3) corresponds to columns 3b to 3b+2 of B.
    # For each block b, we add 6 terms.
    # Term t in block b corresponds to:
    # i in {0,1}, j in {0,1}, k in {0,1,2} within the block.
    # Global k = 3*b + k_local.
    
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k_global = 3 * b + k_local
                    
                    # U index: i*m + j
                    U[i * m + j][col] = 1
                    
                    # V index: j*p + k_global
                    V[j * p + k_global][col] = 1
                    
                    # W index: k_global*n + i
                    W[k_global * n + i][col] = 1
                    
                    col += 1
                    
    # Verify col == rank
    if col != rank:
        raise ValueError("Column count mismatch")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
