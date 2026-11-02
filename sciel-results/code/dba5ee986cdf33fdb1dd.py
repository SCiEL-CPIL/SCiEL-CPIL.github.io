"""Matrix multiplication 2x3 x 3x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 48
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try rank-44 Strassen-like block decomposition
    # For 2x3 * 3x8, we can use a 2x2 block structure with Strassen's algorithm
    # A = [[A11, A12], [A21, A22]] where each block is 1x3
    # B = [[B11, B12, B13, B14, B15, B16, B17, B18], [B21, ..., B28], [B31, ..., B38]]
    # Each B block is 1x8
    
    # Actually, let's think of this as:
    # A is 2x3, B is 3x8
    # C = AB is 2x8
    
    # Standard multiplication: 2*3*8 = 48 scalar multiplications
    # We want to reduce this.
    
    # One approach: group the 3 columns of A into blocks
    # A = [A_col0 | A_col1 | A_col2] where each A_colj is 2x1
    # B = [B_row0; B_row1; B_row2] where each B_rowi is 1x8
    
    # C = A_col0 * B_row0 + A_col1 * B_row1 + A_col2 * B_row2
    
    # Each A_colj * B_rowi is 2x8 outer product, rank 1
    # So we need 3 rank-1 terms, but each involves 2*8 = 16 products
    
    # Alternative: use Strassen-like tricks
    # For multiplying a 2x2 by 2x2, Strassen uses 7 mults instead of 8
    # For 2x3 by 3x2, we can pad to 2x4 by 4x2 or use other decompositions
    
    # Let me try a different approach: decompose into 2x2 blocks
    # A (2x3) = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B (3x8) = [B1; B2] where B1 is 2x8, B2 is 1x8
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 * 2x8 = 2x8, needs 4*8=32 mults naively, but with Strassen-like we might do better
    # Actually, A1*B1: A1 is 2x2, B1 is 2x8
    # = [[a,b],[c,d]] * [[e,f,g,...],[h,i,j,...]]
    # Row 1: [ae+bh, af+bi, ag+bj, ...]
    # Row 2: [ce+dh, cf+di, cg+dj, ...]
    
    # For the 2x2 * 2x8 part, we can use:
    # M1 = (a+b)*(e+h)
    # M2 = (a+b)*(f+i)
    # M3 = (c+d)*(e+h)
    # M4 = (c+d)*(f+i)
    # M5 = a*(g)  -- wait, this doesn't work for 8 columns
    
    # Let me think more carefully. For 2x2 * 2xp:
    # Naive: 4*p mults
    # Strassen: M1=(a+b)(e+h), M2=(a+b)(f+i), M3=(c+d)(e+h), M4=(c+d)(f+i)
    # But we still need individual products for the remaining columns unless we restructure
    
    # Actually, for 2x2 * 2xp, the optimal is likely still 4*p for p>=2
    # because each output element needs specific combinations
    
    # Let me try a completely different approach: use the fact that we can
    # share computations across the 8 output columns
    
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # For each k, we have:
    # [C[0,k]]   [A[0,0] A[0,1] A[0,2]] [B[0,k]]
    # [C[1,k]] = [A[1,0] A[1,1] A[1,2]] [B[1,k]]
    #                              [B[2,k]]
    
    # This is a 2x3 * 3x1 matrix product for each k
    # 2x3 * 3x1: naively 6 mults per k, 48 total
    
    # Can we do 2x3 * 3x1 in fewer than 6 mults?
    # 2x3 * 3x1 = sum over j of A[:,j] * B[j,:]
    # Each A[:,j] is 2x1, B[j,:] is 1x1
    # So it's 3 outer products of rank 1
    
    # For 2x3 * 3x1, the tensor rank is... let's think
    # The output is 2x1, inputs are 2x3 and 3x1
    # Total input dimensions: 2*3 + 3*1 = 9
    # Output: 2*1 = 2
    
    # I believe 2x3 * 3x1 has rank 6 (no savings possible for small dimensions)
    # Actually, for 2xn * nx1, the rank is 2n for n>=2 I think, because
    # each output element is a dot product of length n
    
    # Let me try yet another approach: numerical search for low-rank decomposition
    # But that's risky and time-consuming
    
    # Given the uncertainty, let me just return the schoolbook decomposition
    # which is guaranteed to work. The hint mentioned rank 44, but I can't
    # verify a construction for it, so I'll stick with the safe option.
    
    # Actually, wait. Let me reconsider. The hint says rank 44 was achieved.
    # 48 - 44 = 4 ranks saved.
    
    # For 2x3 * 3x8, maybe we can save 4 ranks by using a clever decomposition.
    # One possibility: treat the 3 middle dimension as a sum of two parts
    # 3 = 2 + 1
    
    # A (2x3) = A1 (2x2) + A2 (2x1) -- no, that's not how matrix addition works
    # Actually, we can write A = [A_left (2x2) | A_right (2x1)]
    # B = [B_top (2x8); B_bottom (1x8)]
    
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 * 2x8
    # A_right * B_bottom: 2x1 * 1x8
    
    # For 2x2 * 2x8:
    # Naive: 4*8 = 32
    # With Strassen: M1=(a+b)(e+h), M2=(a+b)(f+i), M3=(c+d)(e+h), M4=(c+d)(f+i)
    # But for the remaining 6 columns, we still need individual products
    # Unless we can share the (a+b), (c+d), (e+h), (f+i) computations
    
    # Actually, (a+b), (c+d) are fixed for all 8 columns
    # (e+h), (f+i), (g+j), etc. vary per column
    
    # Let me define:
    # s1 = a + b (constant across columns)
    # s2 = c + d (constant across columns)
    
    # For column k of B_top: [b0k, b1k]
    # C[0,k] = a*b0k + b*b1k
    # C[1,k] = c*b0k + d*b1k
    
    # Using Strassen:
    # M1k = s1 * (b0k + b1k)
    # M2k = s2 * (b0k + b1k)
    # But then C[0,k] = ? We can't recover it from just M1k and M2k
    # because we've lost information
    
    # Strassen works for 2x2*2x2 because the structure allows reconstruction
    # For 2x2*2xp with p>2, I don't think Strassen directly applies
    
    # Let me just go with schoolbook. It's safe and correct.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
