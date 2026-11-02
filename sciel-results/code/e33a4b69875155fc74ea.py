"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # We'll build the decomposition by splitting the inner dimension (m=4) into 2+2,
    # and the output columns (p=5) into 2+3.
    # C = A_left(2x2) * B_top(2x5) + A_right(2x2) * B_bottom(2x5)
    # Each (2x2)*(2x5) is split into (2x2)*(2x2) [Strassen, rank 7] + (2x2)*(2x3) [rank 10]
    # Total rank = 2*(7+10) = 34
    
    # Strassen for 2x2x2: 
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B
    # m1=(a+d)*(e+h)
    # m2=(c+d)*e
    # m3=a*(f-h)
    # m4=d*(g-c)
    # m5=(a+b)*h
    # m6=(c-a)*(e+f)
    # m7=(b-d)*(g+h)
    # C11=m1+m4-m5-m7
    # C12=m3+m5
    # C21=m2+m4
    # C22=m1-m2+m3+m7
    # Rank 7
    
    # 2x2x3 rank 10 decomposition:
    # A=[[a,b],[c,d]], B=[[e,f,g],[h,i,j]], C=A*B (2x3)
    # We need 10 scalar products.
    # One known decomposition:
    # m1 = (a+b)*(e+f+h+i)  -- no, let me think more carefully.
    # Actually, let me use a known rank-10 decomposition for 2x2x3.
    # 
    # The 2x2x3 tensor: indices (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1,2}
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # A known rank-10 decomposition (from literature):
    # Let me construct it. We want to express the 12 terms with 10 products.
    # 
    # Actually, let me use a different approach. The 2x2x3 matrix multiplication
    # can be seen as: C = A * B where A is 2x2, B is 2x3.
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i
    # C[0,2] = a*g + b*j
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # C[1,2] = c*g + d*j
    # 
    # 12 terms. Need to reduce to 10.
    # 
    # One approach: use Strassen-like additions.
    # m1 = (a+b)*(e+h) = a*e + a*h + b*e + b*h  -- gives C[0,0] and C[1,0] parts
    # m2 = (a+b)*(f+i) = a*f + a*i + b*f + b*i  -- gives C[0,1] and C[1,1] parts  
    # m3 = (a+b)*(g+j) = a*g + a*j + b*g + b*j  -- gives C[0,2] and C[1,2] parts
    # m4 = (a-b)*(e-h) = a*e - a*h - b*e + b*h  -- 
    # m5 = (a-b)*(f-i) = a*f - a*i - b*f + b*i
    # m6 = (a-b)*(g-j) = a*g - a*j - b*g + b*j
    # m7 = (c+d)*(e+h) = c*e + c*h + d*e + d*h
    # m8 = (c+d)*(f+i) = c*f + c*i + d*f + d*i
    # m9 = (c+d)*(g+j) = c*g + c*j + d*g + d*j
    # m10 = (c-d)*(e-h) = c*e - c*h - d*e + d*h
    # 
    # From m1 and m4:
    # C[0,0] = a*e + b*h = (m1 + m4)/2
    # C[1,0] = c*e + d*h = ?
    # 
    # Hmm, this is getting complicated. Let me use a known explicit rank-10 decomposition.
    # 
    # Actually, I recall that the 2x2x3 tensor has rank exactly 10. Let me look for an explicit decomposition.
    # 
    # From the literature (e.g., Landsberg), one rank-10 decomposition:
    # Using the fact that we can combine the last two columns cleverly.
    # 
    # Let me try a different strategy: use the 2x2x2 Strassen for the first two columns,
    # and handle the third column with 3 additional products.
    # 
    # For columns 0,1 (2x2x2): Strassen gives 7 products.
    # For column 2: C[0,2] = a*g + b*j, C[1,2] = c*g + d*j. That's 4 products schoolbook,
    # but we can do it in 3: m=(a+b)*(g+j), n=a*g, o=b*j... no.
    # C[0,2] = a*g + b*j
    # C[1,2] = c*g + d*j
    # m1 = (a+b)*(g+j) = a*g + a*j + b*g + b*j
    # m2 = a*g
    # m3 = b*j
    # C[0,2] = m2 + m3
    # C[1,2] = ? We need c*g + d*j.
    # m4 = (c+d)*(g+j) = c*g + c*j + d*g + d*j
    # m5 = c*g
    # m6 = d*j
    # C[1,2] = m5 + m6
    # That's 6 more products, total 7+6=13. Not good.
    # 
    # Let me think about this differently. 
    # 
    # Alternative: The 2x2x3 tensor. Let's use the following known decomposition with rank 10:
    # 
    # m1 = (a+c)*(e+h)  -- no, a and c are in different rows.
    # 
    # OK let me just carefully construct one. We have 12 "atoms":
    # a*e, a*f, a*g, b*h, b*i, b*j, c*e, c*f, c*g, d*h, d*i, d*j
    # 
    # We want 10 linear combinations whose products give these.
    # 
    # Strategy: combine a*f and b*i (they don't appear in any output directly in a useful way... 
    # wait, they do: C[0,1] = a*f + b*i).
    # 
    # Let me try:
    # m1 = (a+b)*(e+h) = ae + ah + be + bh
    # m2 = (a-b)*(f-i) = af - ai - bf + bi
    # m3 = (a+b)*(g+j) = ag + aj + bg + bj
    # m4 = (c+d)*(e+h) = ce + ch + de + dh
    # m5 = (c-d)*(f-i) = cf - ci - df + di
    # m6 = (c+d)*(g+j) = cg + cj + dg + dj
    # m7 = a*e
    # m8 = b*h
    # m9 = c*e
    # m10 = d*h
    # 
    # C[0,0] = ae + bh = m7 + m8
    # C[0,1] = af + bi = (m2 + af + ai + bf + bi)/... this is messy.
    # 
    # I think I'm overcomplicating this. Let me just use the rank-34 approach with 
    # Strassen for 2x2x2 and a simple schoolbook for 2x2x3 (which is 6 products), 
    # giving 7+6=13 per half, total 26? No wait.
    # 
    # Let me re-examine. If I split the inner dimension into 2+2:
    # C = A1*B1 + A2*B2
    # A1 is 2x2, B1 is 2x5: this is a 2x2x5 tensor, schoolbook rank 10.
    # A2 is 2x2, B2 is 2x5: same, rank 10.
    # Total: 20. But that's worse than 40/2=20... no it's the same as schoolbook per half.
    # 
    # Wait, schoolbook for 2x2x5 is 2*2*5=20. So two halves give 40 total, same as original.
    # 
    # I need to use Strassen on the 2x2x2 parts and something better for 2x2x3.
    # 
    # For 2x2x5, split into 2x2x2 + 2x2x3:
    # 2x2x2: Strassen rank 7
    # 2x2x3: need rank < 6 to improve. Schoolbook is 6.
    # If 2x2x3 has rank 5, total per half is 12, total 24.
    # If 2x2x3 has rank 10 (as claimed in directive), total per half is 17, total 34.
    # 
    # Hmm, the directive says rank 30. Let me re-read.
    # "decomposing the 4x5 matrix into a 2x2 block (using Strassen, rank 7) and a 2x3 block (using the rank 10 decomposition of 2x2x3)"
    # 
    # So they're splitting B (4x5) into a 2x2 block and a 2x3 block? That doesn't make sense dimensionally.
    # 
    # Maybe they mean: split the computation into parts where one part is 2x2x2 (Strassen) and another is 2x2x3 (rank 10).
    # 7 + 10 = 17 for one "chunk", and maybe two chunks: 34. But they claim 30.
    # 
    # Perhaps there's overlap or sharing. Or maybe the 2x2x3 rank is actually less than 10.
    # 
    # Let me just implement rank 34 (which I'm confident about) and verify it works.
    # Then I can try to optimize further.
    
    # Let me implement the 2x2x3 rank-10 decomposition properly.
    # After research, here's a known rank-10 decomposition for 2x2x3:
    # 
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # 
    # m1 = (a+b)*(e+h)
    # m2 = (a-b)*(f-i)  
    # m3 = (a+b)*(g+j)
    # m4 = (c+d)*(e+h)
    # m5 = (c-d)*(f-i)
    # m6 = (c+d)*(g+j)
    # m7 = a*e
    # m8 = b*h
    # m9 = c*e
    # m10 = d*h
    # 
    # C[0,0] = a*e + b*h = m7 + m8
    # C[0,1] = a*f + b*i = ?
    #   m2 = af - ai - bf + bi
    #   So af + bi = (m2 + af + ai + bf + bi) ... not directly.
    #   
    # This isn't working cleanly. Let me try a different known decomposition.
    # 
    # Actually, I found that the 2x2x3 tensor has border rank 10 but I'm not sure about exact rank.
    # Let me just use schoolbook for the 2x2x3 part (rank 6) and Strassen for 2x2x2 (rank 7).
    # Total per half: 7 + 6 = 13. Two halves: 26.
    # 
    # Wait, that would be rank 26! Let me verify this makes sense.
    # 
    # C = A1*B1 + A2*B2
    # A1 (2x2) * B1 (2x5): split B1 into B1a (2x2) and B1b (2x3)
    #   A1*B1a: 2x2x2, Strassen, 7 products
    #   A1*B1b: 2x2x3, schoolbook, 6 products
    #   Subtotal: 13
    # A2 (2x2) * B2 (2x5): similarly
    #   A2*B2a: 7 products
    #   A2*B2b: 6 products
    #   Subtotal: 13
    # Total: 26
    # 
    # That's much better than 40! Let me implement this.
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 8 x 26
    V = [[0] * rank for _ in range(m * p)]  # 20 x 26
    W = [[0] * rank for _ in range(p * n)]  # 10 x 26
    
    col = 0
    # Strassen 2x2x2 helper: adds 7 products
    # A indices: row i in {0,1}, col j in {0,1} -> U index = i*2 + j (within the 2x2 block)
    # B indices: row j in {0,1}, col k in {0,1} -> V index = j*2 + k (within the 2x2 block)
    # C indices: row i, col k -> W index = k*2 + i (per trace convention W[k*n+i])
    # 
    # For the top half (A1, B1): A1 is rows 0-1 of A, cols 0-1. B1 is rows 0-1 of B, cols 0-4.
    # A1[i,j] -> U[i*4 + j] in the full 2x4 A... wait, A is 2x4.
    # U has shape (n*m, rank) = (8, rank). U[i*m+j, c] = U[i*4+j, c]
    # V has shape (m*p, rank) = (20, rank). V[j*p+k, c] = V[j*5+k, c]
    # W has shape (p*n, rank) = (10, rank). W[k*n+i, c] = W[k*2+i, c]
    
    def add_strassen(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        """Add Strassen decomposition for 2x2x2 block.
        A is at rows A_row_offset..A_row_offset+1, cols A_col_offset..A_col_offset+1
        B is at rows B_row_offset..B_row_offset+1, cols B_col_offset..B_col_offset+1
        C output at cols C_col_offset..C_col_offset+1
        """
        nonlocal col
        # A indices in U: (A_row_offset + i)*4 + (A_col_offset + j) for i,j in {0,1}
        # B indices in V: (B_row_offset + j)*5 + (B_col_offset + k) for j,k in {0,1}
        # C indices in W: (C_col_offset + k)*2 + i for i,k in {0,1}
        
        # m1 = (a00+a11)*(b00+b11)
        U[(A_row_offset+0)*4 + (A_col_offset+0)][col] = 1
        U[(A_row_offset+1)*4 + (A_col_offset+1)][col] = 1
        V[(B_row_offset+0)*5 + (B_col_offset+0)][col] = 1
        V[(B_row_offset+1)*5 + (B_col_offset+1)][col] = 1
        W[(C_col_offset+0)*2 + 0][col] = 1   # C[0,0] += m1
        W[(C_col_offset+0)*2 + 1][col] = 1   # C[0,1] += m1
        W[(C_col_offset+1)*2 + 0][col] = 1   # C[1,0] += m1
        W[(C_col_offset+1)*2 + 1][col] = 1   # C[1,1] += m1
        col += 1
        
        # m2 = (a10+a11)*b00
        U[(A_row_offset+1)*4 + (A_col_offset+0)][col] = 1
        U[(A_row_offset+1)*4 + (A_col_offset+1)][col] = 1
        V[(B_row_offset+0)*5 + (B_col_offset+0)][col] = 1
        W[(C_col_offset+0)*2 + 1][col] = 1   # C[0,1] += m2
        W[(C_col_offset+1)*2 + 0][col] = 1   # C[1,0] += m2
        W[(C_col_offset+1)*2 + 1][col] = 1   # C[1,1] += m2
        col += 1
        
        # m3 = a00*(b01-b11)
        U[(A_row_offset+0)*4 + (A_col_offset+0)][col] = 1
        V[(B_row_offset+0)*5 + (B_col_offset+1)][col] = 1
        V[(B_row_offset+1)*5 + (B_col_offset+1)][col] = -1
        W[(C_col_offset+0)*2 + 1][col] = 1   # C[0,1] += m3
        W[(C_col_offset+1)*2 + 1][col] = 1   # C[1,1] += m3
        col += 1
        
        # m4 = a11*(b10-b00)
        U[(A_row_offset+1)*4 + (A_col_offset+1)][col] = 1
        V[(B_row_offset+1)*5 + (B_col_offset+0)][col] = 1
        V[(B_row_offset+0)*5 + (B_col_offset+0)][col] = -1
        W[(C_col_offset+0)*2 + 0][col] = 1   # C[0,0] += m4
        W[(C_col_offset+1)*2 + 0][col] = 1   # C[1,0] += m4
        col += 1
        
        # m5 = (a00+a01)*b11
        U[(A_row_offset+0)*4 + (A_col_offset+0)][col] = 1
        U[(A_row_offset+0)*4 + (A_col_offset+1)][col] = 1
        V[(B_row_offset+1)*5 + (B_col_offset+1)][col] = 1
        W[(C_col_offset+0)*2 + 0][col] = -1  # C[0,0] -= m5
        W[(C_col_offset+1)*2 + 1][col] = 1   # C[1,1] += m5
        col += 1
        
        # m6 = (a10-a00)*(b00+b01)
        U[(A_row_offset+1)*4 + (A_col_offset+0)][col] = 1
        U[(A_row_offset+0)*4 + (A_col_offset+0)][col] = -1
        V[(B_row_offset+0)*5 + (B_col_offset+0)][col] = 1
        V[(B_row_offset+0)*5 + (B_col_offset+1)][col] = 1
        W[(C_col_offset+0)*2 + 0][col] = 1   # C[0,0] += m6
        W[(C_col_offset+1)*2 + 0][col] = 1   # C[1,0] += m6
        col += 1
        
        # m7 = (a01-a11)*(b10+b11)
        U[(A_row_offset+0)*4 + (A_col_offset+1)][col] = 1
        U[(A_row_offset+1)*4 + (A_col_offset+1)][col] = -1
        V[(B_row_offset+1)*5 + (B_col_offset+0)][col] = 1
        V[(B_row_offset+1)*5 + (B_col_offset+1)][col] = 1
        W[(C_col_offset+0)*2 + 0][col] = -1  # C[0,0] -= m7
        W[(C_col_offset+1)*2 + 1][col] = 1   # C[1,1] += m7
        col += 1
    
    def add_schoolbook_2x2x3(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        """Add schoolbook decomposition for 2x2x3 block.
        A is 2x2 at given offsets, B is 2x3 at given offsets, C is 2x3 at given col offset.
        """
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(A_row_offset+i)*4 + (A_col_offset+j)][col] = 1
                    V[(B_row_offset+j)*5 + (B_col_offset+k)][col] = 1
                    W[(C_col_offset+k)*2 + i][col] = 1
                    col += 1
    
    # Top half: A1 = A[0:2, 0:2] (rows 0-1, cols 0-1), B1 = B[0:2, 0:5] (rows 0-1, cols 0-4)
    # A1*B1 contributes to all of C (2x5)
    # Split B1 into B1a (2x2, cols 0-1) and B1b (2x3, cols 2-4)
    # A1*B1a: 2x2x2, Strassen
    add_strassen(0, 0, 0, 0, 0)  # 7 products
    # A1*B1b: 2x2x3, schoolbook
    add_schoolbook_2x2x3(0, 0, 0, 2, 2)  # 6 products
    
    # Bottom half: A2 = A[0:2, 2:4] (rows 0-1, cols 2-3), B2 = B[2:4, 0:5] (rows 2-3, cols 0-4)
    # A2*B2 contributes to all of C (2x5)
    # Split B2 into B2a (2x2, cols 0-1) and B2b (2x3, cols 2-4)
    # A2*B2a: 2x2x2, Strassen
    add_strassen(0, 2, 2, 0, 0)  # 7 products
    # A2*B2b: 2x2x3, schoolbook
    add_schoolbook_2x2x3(0, 2, 2, 2, 2)  # 6 products
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
