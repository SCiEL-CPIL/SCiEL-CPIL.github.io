def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    
    # We split the multiplication C = A B into two parts:
    # C = A_1 B_1 + A_2 B_2
    # A_1 is 2x2 (cols 1,2 of A), B_1 is 2x8 (rows 1,2 of B)
    # A_2 is 2x1 (col 3 of A), B_2 is 1x8 (row 3 of B)
    
    # Part 1: 2x2 by 2x8. Split into four 2x2 by 2x2 blocks.
    # Rank = 4 * 7 = 28.
    # Part 2: 2x1 by 1x8. Rank = 2 * 8 = 16.
    # Total Rank = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 2x2 * 2x8
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..7
    # C indices: k in 0..7, i in 0..1
    
    # We use 4 blocks of Strassen for 2x2*2x2
    # Strassen factors for 2x2*2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # We will map these to the global U, V, W.
    # For each block t (0..3), columns k are 2t, 2t+1.
    # j indices are 0, 1.
    # i indices are 0, 1.
    
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # 7 multiplications for this block
        # m1
        U[0][col] = 1; U[3][col] = 1 # a00, a11 (i=0,j=0; i=1,j=1) -> U[i*m+j]
        V[0][col] = 1; V[1][col] = 1 # b00, b11 (j=0,k=k0; j=1,k=k1) -> V[j*p+k]
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1 # c00, c11? No, W[k*n+i]
        # Wait, Strassen formulas need to be mapped carefully.
        # c00 = m1 + m4 - m5 - m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m7
        
        # Let's just use the naive 4 multiplications for 2x2*2x2?
        # Rank 4 vs 7. 4 is worse.
        # We need the Strassen decomposition.
        
        # Let's define the 7 terms explicitly.
        # Term 1: m1
        U[0][col] += 1; U[3][col] += 1
        V[0][col] += 1; V[1][col] += 1
        W[k0*n+0][col] += 1 # c00
        W[k1*n+1][col] += 1 # c11
        col += 1
        
        # Term 2: m2 = (a10+a11)b00
        U[2][col] += 1; U[3][col] += 1 # a10 (i=1,j=0), a11 (i=1,j=1)
        V[0][col] += 1 # b00
        W[k0*n+1][col] += 1 # c10
        col += 1
        
        # Term 3: m3 = a00(b10-b11)
        U[0][col] += 1 # a00
        V[1][col] += 1; V[2][col] += -1 # b10 (j=1,k=k0), b11 (j=1,k=k1)
        W[k0*n+0][col] += 1 # c00
        W[k1*n+1][col] += -1 # c11? No, c11 has +m7, c00 has -m7?
        # Let's re-verify Strassen.
        # c00 = m1 + m4 - m5 - m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m7
        
        # My previous mapping for m1 was c00 and c11.
        # m1 contributes +1 to c00 and +1 to c11.
        # m2 contributes +1 to c10.
        # m3 contributes +1 to c01 and +1 to c11.
        # m4 contributes +1 to c00 and +1 to c10.
        # m5 contributes -1 to c00 and +1 to c01.
        # m6 is not used in standard Strassen?
        # Standard Strassen uses 7 mults.
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        
        # c00 = m1 + m4 - m5 - m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m7
        
        # Let's re-do the columns.
        # I already added m1.
        # m1: U[0]+=1, U[3]+=1; V[0]+=1, V[1]+=1; W[k0,0]+=1, W[k1,1]+=1.
        
        # m2: (a10+a11)b00
        # U[2]+=1, U[3]+=1; V[0]+=1; W[k0,1]+=1 (c10)
        # But c10 = m2 + m4. So W[k0,1] gets +1 from m2.
        
        # m3: a00(b10-b11)
        # U[0]+=1; V[1]+=1, V[2]+=-1; W[k1,0]+=1 (c01), W[k1,1]+=1 (c11)
        # Wait, c01 is W[k1, 0]? No.
        # C is 2x8. i=0,1. k=0..7.
        # c01 corresponds to i=0, k=k1? No.
        # In the 2x2 block, local indices are i_loc, k_loc.
        # i_loc=0, k_loc=0 -> c00 -> global i=0, k=k0.
        # i_loc=0, k_loc=1 -> c01 -> global i=0, k=k1.
        # i_loc=1, k_loc=0 -> c10 -> global i=1, k=k0.
        # i_loc=1, k_loc=1 -> c11 -> global i=1, k=k1.
        
        # So:
        # c00 -> W[k0*n+0]
        # c01 -> W[k1*n+0]
        # c10 -> W[k0*n+1]
        # c11 -> W[k1*n+1]
        
        # Let's restart the block construction with correct W indices.
        col -= 1 # rollback m1
        # m1
        U[0][col] += 1; U[3][col] += 1
        V[0][col] += 1; V[1][col] += 1
        W[k0*n+0][col] += 1 # c00
        W[k1*n+1][col] += 1 # c11
        col += 1
        
        # m2
        U[2][col] += 1; U[3][col] += 1
        V[0][col] += 1
        W[k0*n+1][col] += 1 # c10
        col += 1
        
        # m3
        U[0][col] += 1
        V[1][col] += 1; V[2][col] += -1
        W[k1*n+0][col] += 1 # c01
        W[k1*n+1][col] += 1 # c11
        col += 1
        
        # m4
        U[3][col] += 1
        V[2][col] += 1; V[0][col] += -1 # b01 (j=0,k=k1), b00 (j=0,k=k0)
        W[k0*n+0][col] += 1 # c00
        W[k0*n+1][col] += 1 # c10
        col += 1
        
        # m5
        U[0][col] += 1; U[1][col] += 1 # a00, a01
        V[1][col] += 1 # b11 (j=1,k=k1)
        W[k0*n+0][col] += -1 # c00
        W[k1*n+0][col] += 1 # c01
        col += 1
        
        # m6
        U[2][col] += 1; U[0][col] += -1 # a10, a00
        V[0][col] += 1; V[1][col] += 1 # b00, b10
        # m6 is not in the final sum?
        # Standard Strassen:
        # c00 = m1 + m4 - m5 - m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m7
        # m6 is not used in the output?
        # Wait, standard Strassen uses 7 mults.
        # m1..m7.
        # Is m6 used?
        # Some definitions use m6 = (a10-a00)(b00+b10) and it is not in the output?
        # No, that would be wasteful.
        # Let's check a reliable source for Strassen 7 mults.
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        # c00 = m1 + m4 - m5 - m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m7
        # Yes, m6 is not in the output. This implies my set of mults is wrong or I'm missing a term.
        # Actually, c10 = m2 + m4.
        # c00 = m1 + m4 - m5 - m7.
        # It seems m6 is indeed not used in this specific formulation?
        # No, that can't be right. Rank 7 means 7 mults are used.
        # Ah, the formulation above uses m1-m5, m7. That's 6.
        # There is a 7th one.
        # Let's use a different known set.
        # Or just use naive 4 for this sub-block?
        # If I use naive 4, rank is 4*4 + 16 = 32.
        # 32 is better than 44!
        # Why did I think Strassen was 7? It is 7 for 2x2x2.
        # Naive is 4. 7 > 4.
        # Strassen reduces the *number of multiplications* for large matrices by recursion.
        # For a single 2x2x2 block, naive is 4. Strassen is 7.
        # So naive is better for a single 2x2x2!
        # Strassen is only beneficial when you have to do many 2x2x2s and the overhead is amortized?
        # No, Strassen reduces the exponent.
        # But for a fixed 2x2x2, rank is 7. Naive rank is 4.
        # So we should use NAIVE for the 2x2x2 blocks!
        
        # So for 2x2 * 2x8, we have four 2x2 * 2x2 blocks.
        # Each has rank 4.
        # Total rank 16.
        # Plus 2x1 * 1x8 has rank 16.
        # Total rank 32.
        
        # This is much better.
        # Let's implement rank 32.
        
        col -= 5 # rollback m2-m5
        # We will just use naive for the 2x2 blocks.
        
    # Reset col to 0 and implement naive for Part 1 and Part 2.
    col = 0
    # Part 1: 2x2 * 2x8 (Naive)
    # i in 0..1, j in 0..1, k in 0..7
    for i in range(2):
        for j in range(2):
            for k in range(8):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    # col is now 16.
    
    # Part 2: 2x1 * 1x8 (Naive)
    # i in 0..1, j=2, k in 0..7
    for i in range(2):
        for k in range(8):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    # col is now 32.
    
    return {"U": U, "V": V, "W": W}
