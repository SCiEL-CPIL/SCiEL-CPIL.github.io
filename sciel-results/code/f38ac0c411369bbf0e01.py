"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition.
    # Split 4x4 into 2x2 blocks.
    # Rank of 2x2x2 is 7 (Strassen).
    
    # Strassen factors for 2x2x2
    # T = sum_{t=1}^7 u_t \otimes v_t \otimes w_t
    # u, v, w are 2x2 matrices.
    # C = sum_t (u_t^T A) (v_t^T B) w_t^T ? No.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # Strassen:
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
    
    # We need the rank-1 decomposition of the 2x2x2 tensor.
    # The 7 terms above are NOT rank-1 in the tensor sense of A and B.
    # They are products of linear combinations.
    # Each M_t is a 2x2 matrix product.
    # We can decompose each M_t into 4 schoolbook terms?
    # No, we want the rank of the 2x2x2 tensor to be 7.
    # The 7 terms M1..M7 are the 7 multiplications.
    # Each multiplication is a scalar multiplication of two 2x2 matrices?
    # No, M1 is a 2x2 matrix.
    # M1 = X1 Y1 where X1, Y1 are 2x2.
    # The tensor for 2x2x2 is the sum of 7 rank-1 tensors?
    # No, the rank is 7.
    # The 7 terms in Strassen are the rank-1 components?
    # Yes, if we consider the "multiplication" as a bilinear map.
    # But M1 is a matrix.
    # We need to express C = sum_t (linear in A) * (linear in B) * (linear in C)?
    # No, C is the output.
    # The tensor is T.
    # T(A,B) = C.
    # Strassen says T = sum_{t=1}^7 T_t where T_t are rank-1?
    # No, T_t are not rank-1.
    # The rank of the 2x2x2 tensor is 7.
    # This means it can be written as sum of 7 rank-1 tensors.
    # I will use the standard Strassen factors for the 2x2x2 tensor.
    
    # Factors for 2x2x2 (from literature):
    # u1 = [[1,0],[1,1]], v1 = [[1,1],[0,1]], w1 = [[1,1],[1,0]]
    # ... I need the exact 21 factors? No, 7 factors for 2x2.
    
    # Let's use a simpler approach:
    # Just use Schoolbook for everything.
    # Rank 96.
    # But I want to improve.
    
    # Let's use the block decomposition with Schoolbook for blocks.
    # Rank 8x4 (for 4x4x4) + 4x4 (for 4x4x2) = 32 + 16 = 48?
    # No.
    # 4x4x4: 8 block mults. Each 2x2x2 schoolbook rank 4. Total 32.
    # 4x4x2: 4 block mults. Each 2x2x2 schoolbook rank 4. Total 16.
    # Total 48.
    
    # This is very easy to implement.
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4x4x4 part
    for r in range(2):
        for c in range(2):
            for k in range(2):
                # Block A[r, k], B[k, c], C[r, c]
                # Schoolbook for 2x2
                for i in range(2):
                    for j in range(2):
                        # A index: (r*2 + i, k*2 + j)
                        # B index: (k*2 + j, c*2 + i) ? No.
                        # B is 4x6.
                        # For 4x4x4 part, B cols 0..3.
                        # B block [k, c] is rows k*2..k*2+1, cols c*2..c*2+1.
                        # C block [r, c] is rows r*2..r*2+1, cols c*2..c*2+1.
                        
                        # Term: A_{r*2+i, k*2+j} * B_{k*2+j, c*2+i} -> C_{r*2+i, c*2+i}
                        # Wait, C index is (r*2+i, c*2+l)?
                        # C_{i, l} = sum_j A_{i, j} B_{j, l}.
                        # Block C_{r,c} corresponds to i in r*2..r*2+1, l in c*2..c*2+1.
                        # B_{j, l} where j in k*2..k*2+1, l in c*2..c*2+1.
                        # A_{i, j} where i in r*2..r*2+1, j in k*2..k*2+1.
                        
                        # So for fixed r, c, k:
                        # We need to decompose the 2x2x2 tensor for this block.
                        # Schoolbook: 4 terms.
                        for l in range(2):
                            # i = r*2 + i_local
                            # j = k*2 + j_local
                            # l = c*2 + l_local
                            # B index row is j, col is l.
                            # A index row is i, col is j.
                            # C index row is i, col is l.
                            
                            i_loc = 0 # We iterate i_loc, j_loc, l_loc?
                            # No, we iterate over the 4 terms of 2x2 schoolbook.
                            # Term (i_loc, j_loc, l_loc) where i_loc, j_loc, l_loc in 0..1?
                            # No, schoolbook for 2x2 is:
                            # C00 = A00B00 + A01B10
                            # C01 = A00B01 + A01B11
                            # C10 = A10B00 + A11B10
                            # C11 = A10B01 + A11B11
                            # There are 8 terms.
                            # So rank is 8.
                            
                            pass
                            
    # This is getting complicated.
    # I will return the schoolbook rank 96 to be safe?
    # No, the directive wants improvement.
    # I will use the rank 84 (Strassen blocks) if I can.
    # But I don't have the factors.
    
    # I will use the rank 96 schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
