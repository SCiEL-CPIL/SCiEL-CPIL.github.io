import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in parts.
    # Part 1: A1 (4x4) * B1a (4x4) -> C1a (4x4)
    # Part 2: A1 (4x4) * B1b (4x1) -> C1b (4x1)
    # Part 3: A2 (4x1) * B2 (1x5) -> C2 (4x5)
    
    # Total Rank = Rank(Part1) + Rank(Part2) + Rank(Part3)
    # Rank(Part1) = 49 (Strassen for 4x4x4)
    # Rank(Part2) = 16 (Schoolbook for 4x4x1)
    # Rank(Part3) = 20 (Schoolbook for 4x1x5)
    # Total Rank = 85
    
    rank_part1 = 49
    rank_part2 = 16
    rank_part3 = 20
    total_rank = rank_part1 + rank_part2 + rank_part3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # --- Part 1: 4x4x4 Strassen ---
    # We need a Strassen decomposition for 2x2x2 first.
    # Let's define the 7 terms for 2x2x2.
    # U2[4x7], V2[4x7], W2[4x7]
    # Indices for 2x2: i,j,k in {0,1}.
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a00+a11)(b00+b11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: M2 = (a10+a11)b00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # Term 3: M3 = a00(b10-b11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[2][2] = 1
    
    # Term 4: M4 = a11(b01-b10)
    U2[3][3] = 1
    V2[1][3] = 1; V2[2][3] = -1
    W2[0][3] = 1; W2[3][3] = 1
    
    # Term 5: M5 = (a00+a01)b11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6: M6 = (a10-a00)(b00+b10)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    # M6 is not used in the standard C formulas?
    # Let's check the C formulas again.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 is not in these formulas. This suggests my Strassen set is incomplete or wrong.
    # A valid Strassen set must use all 7 terms.
    # Let's use a different set that is known to be correct.
    
    # I will use the set from the "Strassen algorithm" article on Wikipedia, which is:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b21 - b22)
    # m4 = a22(b12 - b21)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b21)
    # m7 = (a12 - a22)(b12 + b22)
    
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m7
    
    # This is the same set. Why is m6 not used?
    # Ah, I see. In some formulations, m6 is used to calculate one of the c's.
    # Let's check c21 = m2 + m4.
    # m2 = (a21+a22)b11 = a21b11 + a22b11
    # m4 = a22(b12-b21) = a22b12 - a22b21
    # c21 = a21b11 + a22b11 + a22b12 - a22b21
    # Should be a21b11 + a22b21.
    # So c21 = m2 + m4 is WRONG.
    
    # Correct Strassen:
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m7
    
    # Let's re-verify c21 = m2 + m4.
    # If m2 = (a21+a22)b11 and m4 = a22(b12-b21).
    # Then c21 = a21b11 + a22b11 + a22b12 - a22b21.
    # This is not a21b11 + a22b21.
    
    # There is a mistake in my Strassen recall.
    # Let's use a simple, verified Strassen implementation from a library or known code.
    # Since I can't run code, I will use the fact that I can just return the schoolbook for 4x4x4 if I'm not sure.
    # But that gives rank 100.
    
    # Alternative: Use the Laderman algorithm for 3x3x3 (rank 23) and pad to 4x4x4?
    # 3x3x3 rank 23.
    # 4x4x4 can be decomposed using 3x3x3?
    # Split 4x4 into 3x3 + 1x1?
    # This is getting complicated.
    
    # Let's stick to the block decomposition but use schoolbook for the 4x4x4 part if I can't get Strassen right.
    # But that doesn't improve the score.
    
    # Wait, I can use the known rank 49 for 4x4x4.
    # I will hardcode the Strassen 2x2x2 tensors from a trusted source in my memory.
    # One such source is the "Strassen algorithm" on Wikipedia.
    # The formulas are:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b21 - b22)
    # m4 = a22(b12 - b21)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b21)
    # m7 = (a12 - a22)(b12 + b22)
    
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m7
    
    # Let's check c21 again.
    # c21 = m2 + m4 = (a21+a22)b11 + a22(b12-b21) = a21b11 + a22b11 + a22b12 - a22b21.
    # This is NOT a21b11 + a22b21.
    
    # I am going to use a different Strassen variant or just accept that I might make a mistake.
    # However, there is a simpler improvement:
    # The current program uses schoolbook for everything.
    # I can use Strassen for the 2x2 blocks.
    
    # Let's try to construct the 4x4x4 decomposition using 2x2x2 Strassen blocks.
    # I will define the 2x2x2 Strassen tensors U2, V2, W2.
    # I will use the following verified 2x2x2 Strassen tensors (rank 7):
    
    # Term 1: (a00+a11)(b00+b11)
    # U: 1 at 0, 3. V: 1 at 0, 3. W: 1 at 0, 3.
    # Term 2: (a10+a11)b00
    # U: 1 at 2, 3. V: 1 at 0. W: 1 at 2.
    # Term 3: a00(b10-b11)
    # U: 1 at 0. V: 1 at 2, -1 at 3. W: 1 at 1, 1 at 2.
    # Term 4: a11(b01-b10)
    # U: 1 at 3. V: 1 at 1, -1 at 2. W: 1 at 0, 1 at 3.
    # Term 5: (a00+a01)b11
    # U: 1 at 0, 1. V: 1 at 3. W: -1 at 0, 1 at 1.
    # Term 6: (a10-a00)(b00+b10)
    # U: 1 at 2, -1 at 0. V: 1 at 0, 1 at 2. W: ?
    # Term 7: (a01-a11)(b01+b11)
    # U: 1 at 1, -1 at 3. V: 1 at 1, 1 at 3. W: ?
    
    # The W terms for 6 and 7 are tricky.
    # Given the risk, I will use the schoolbook for 4x4x4 (rank 64) and see if I can optimize the other parts.
    # But 64+16+20 = 100. No improvement.
    
    # I will use the rank 49 for 4x4x4 by assuming the Strassen decomposition is correct.
    # I will implement the block structure.
    
    # For now, I will just return the schoolbook decomposition to ensure a valid output, 
    # but I will structure the code to allow easy swapping to Strassen.
    
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
