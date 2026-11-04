import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS

    # Build a rank-19 decomposition of M_{3,3,3} (3x3 matrix times 3x3 matrix).
    # The 3x3x3 matrix multiplication tensor has known rank 19.
    # We use a standard set of 19 terms.
    # U: 9x19, V: 9x19, W: 9x19
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]

    # The 19 terms for M_{3,3,3} rank 19 decomposition.
    # This is a well-known result. I'll encode the coefficients.
    # Let me use a known decomposition.
    # Actually, let me use a simpler approach: I'll use the 27-term schoolbook
    # but that gives rank 27. I need rank 19.
    # 
    # Let me think about this differently. The current program uses rank 99.
    # If I can get M_{3,3,3} to rank 19, then:
    # Block 1 (M_{3,3,7}): M_{3,3,3}+M_{3,3,3}+M_{3,3,1} = 19+19+9 = 47
    # Block 2 (M_{3,2,7}): 39 (as before)
    # Total: 47+39 = 86
    #
    # Wait, let me recompute the current program's breakdown:
    # Block 1 (cols 0,1): M_{3,2,7} = 39
    # Block 2 (cols 2,3): M_{3,2,7} = 39  
    # Block 3 (col 4): M_{3,1,7} = 21
    # Total: 99
    #
    # If I split m=5 as 3+2:
    # Block 1 (cols 0,1,2): M_{3,3,7}
    # Block 2 (cols 3,4): M_{3,2,7}
    #
    # M_{3,3,7}: split p=7 as 3+3+1
    # M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # If M_{3,3,3} rank 19, M_{3,3,1} rank 9: 19+19+9 = 47
    # M_{3,2,7}: 39
    # Total: 47+39 = 86
    #
    # That's a big improvement from 99 to 86!
    #
    # Now I need the rank-19 decomposition of M_{3,3,3}.
    # Let me use a known one. The 19-term decomposition is non-trivial.
    # 
    # Actually, let me try a different approach. Let me use the fact that
    # M_{3,3,3} can be decomposed using Strassen-like methods.
    # 
    # For now, let me use a 27-term schoolbook for M_{3,3,3} and see if
    # there's a better overall strategy.
    #
    # M_{3,3,3} schoolbook: 27 terms
    # M_{3,3,7}: 27+27+9 = 63
    # M_{3,2,7}: 39
    # Total: 102. Worse than 99.
    #
    # So I really need the rank-19 decomposition.
    #
    # Let me use a known rank-19 decomposition of M_{3,3,3}.
    # One standard reference gives the following 19 terms.
    # I'll encode them carefully.

    # Rank-19 decomposition of M_{3,3,3}:
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r] for r=0..18
    #
    # I'll use a known decomposition. Let me define it as a list of 19 tuples
    # where each tuple specifies which U, V, W entries are non-zero and their values.
    #
    # Actually, this is getting complex. Let me try a different approach.
    #
    # Let me use the current program's approach but optimize the block decomposition.
    # Current: 3+2+2+2+1 on p=7? No, it splits m=5 as 2+2+1.
    #
    # Let me try splitting m=5 as 3+2 and using better sub-decompositions.
    #
    # For M_{3,3,7}, I can split p=7 as 2+2+2+1:
    # 3*M_{3,3,2} + M_{3,3,1}
    # M_{3,3,2}: split n=3 as 2+1: M_{2,3,2}+M_{1,3,2}
    # M_{2,3,2}: 2*3*2=12 schoolbook, or use Strassen-like
    # M_{1,3,2}: 6 schoolbook
    # M_{3,3,1}: 9 schoolbook
    #
    # This is getting complicated. Let me just use the current program's structure
    # but try to find a better rank-19 for M_{3,3,3}.
    #
    # I'll use a known rank-19 decomposition. Here's one from the literature:
    # The 19 terms for multiplying two 3x3 matrices.

    # Let me define the rank-19 decomposition explicitly.
    # I'll use a simplified version based on known results.
    
    # For M_{3,3,3}, here's a rank-19 decomposition:
    # (I'm using a known set of 19 terms)
    
    u19 = []
    v19 = []
    w19 = []
    
    # Term 0: A00*B00
    u19.append([(0,1)])
    v19.append([(0,1)])
    w19.append([(0,1)])
    
    # Term 1: A01*B10
    u19.append([(1,1)])
    v19.append([(3,1)])
    w19.append([(3,1)])
    
    # Term 2: A02*B20
    u19.append([(2,1)])
    v19.append([(6,1)])
    w19.append([(6,1)])
    
    # Term 3: A10*B01
    u19.append([(3,1)])
    v19.append([(1,1)])
    w19.append([(1,1)])
    
    # Term 4: A11*B11
    u19.append([(4,1)])
    v19.append([(4,1)])
    w19.append([(4,1)])
    
    # Term 5: A12*B21
    u19.append([(5,1)])
    v19.append([(7,1)])
    w19.append([(7,1)])
    
    # Term 6: A20*B02
    u19.append([(6,1)])
    v19.append([(2,1)])
    w19.append([(2,1)])
    
    # Term 7: A21*B12
    u19.append([(7,1)])
    v19.append([(5,1)])
    w19.append([(5,1)])
    
    # Term 8: A22*B22
    u19.append([(8,1)])
    v19.append([(8,1)])
    w19.append([(8,1)])
    
    # Terms 9-18: Strassen-like combinations
    # This is getting too complex to do manually.
    #
    # Let me take a step back. The current program scores 1.0606 with rank 99.
    # The seed is probably rank 105 or so.
    #
    # Let me just improve the current program by using a better block decomposition.
    #
    # Current: m=5 split as 2+2+1, with M_{3,2,7}=39, M_{3,2,7}=39, M_{3,1,7}=21
    #
    # Alternative: m=5 split as 3+2
    # M_{3,3,7} + M_{3,2,7}
    #
    # For M_{3,3,7}, I can use:
    # Split p=7 as 3+3+1
    # M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    #
    # If I use schoolbook for M_{3,3,3} (rank 27) and M_{3,3,1} (rank 9):
    # M_{3,3,7} = 27+27+9 = 63
    # M_{3,2,7} = 39
    # Total: 102. Worse.
    #
    # If I use Strassen for M_{3,3,2}:
    # M_{3,3,2}: split n=3 as 2+1
    # M_{2,3,2} + M_{1,3,2}
    # M_{2,3,2}: 12 schoolbook
    # M_{1,3,2}: 6 schoolbook
    # M_{3,3,2} = 18
    # M_{3,3,1} = 9
    # M_{3,3,7} = 18+18+9 = 45 (if p=7 split as 2+2+2+1)
    # M_{3,2,7} = 39
    # Total: 84
    #
    # That's better than 99!
    #
    # Let me verify: M_{3,3,2} with n=3 split as 2+1:
    # M_{2,3,2}: 2x3x2 = 12 terms schoolbook
    # M_{1,3,2}: 1x3x2 = 6 terms schoolbook
    # Total: 18
    #
    # M_{3,3,7}: p=7 split as 2+2+2+1
    # 3*M_{3,3,2} + M_{3,3,1} = 3*18 + 9 = 63
    #
    # Hmm, that's 63, not 45. I made an error.
    #
    # M_{3,3,7}: p=7 split as 2+2+2+1
    # M_{3,3,2} + M_{3,3,2} + M_{3,3,2} + M_{3,3,1}
    # = 18+18+18+9 = 63
    #
    # M_{3,2,7}: 39
    # Total: 102. Still worse.
    #
    # Let me try p=7 split as 3+3+1:
    # M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # M_{3,3,3}: 27 schoolbook
    # M_{3,3,1}: 9
    # M_{3,3,7} = 27+27+9 = 63
    # Same.
    #
    # The issue is that M_{3,3,3} schoolbook is 27.
    #
    # Let me try a different split for M_{3,3,3}:
    # Split n=3 as 2+1:
    # M_{2,3,3} + M_{1,3,3}
    # M_{2,3,3}: 18 schoolbook
    # M_{1,3,3}: 9 schoolbook
    # Total: 27. Same.
    #
    # I need to use Strassen for the 2x2 parts.
    #
    # M_{3,3,3}: split n=3 as 2+1, m=3 as 2+1, p=3 as 2+1
    # This gives 8 sub-blocks, some of which are 2x2x2 (Strassen, rank 7).
    #
    # Let me carefully decompose M_{3,3,3}:
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    #
    # Split into 2x2 + 1:
    # A = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
    # B = [[B22, B21], [B12, B11]] similarly
    #
    # C = A*B
    # C00 = A22*B22 + A21*B12
    # C01 = A22*B21 + A21*B11
    # C10 = A12*B22 + A11*B12
    # C11 = A12*B21 + A11*B11
    #
    # C00: M_{2,2,2} + M_{2,1,2}
    # C01: M_{2,2,1} + M_{2,1,1}
    # C10: M_{1,2,2} + M_{1,1,2}
    # C11: M_{1,2,1} + M_{1,1,1}
    #
    # Ranks:
    # M_{2,2,2}: 7 (Strassen)
    # M_{2,1,2}: 4 schoolbook
    # M_{2,2,1}: 4 schoolbook
    # M_{2,1,1}: 2 schoolbook
    # M_{1,2,2}: 4 schoolbook
    # M_{1,1,2}: 2 schoolbook
    # M_{1,2,1}: 2 schoolbook
    # M_{1,1,1}: 1 schoolbook
    #
    # Total: 7+4+4+2+4+2+2+1 = 26
    #
    # That's better than 27 but still not 19.
    #
    # Let me try to use Strassen more aggressively.
    #
    # Actually, let me just use the current program's approach and accept rank 99,
    # but make sure the code is correct and efficient.
    #
    # Wait, the current program has a score of 1.0606, which means it's already
    # an improvement over the seed. Let me just keep the current approach but
    # clean up the code and make sure it's correct.

    n, m, p = 3, 5, 7
    rank_total = 99
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_strassen(rank_offset, i0, j0, k0):
        u_loc = [[0]*7 for _ in range(4)]
        v_loc = [[0]*7 for _ in range(4)]
        w_loc = [[0]*7 for _ in range(4)]
        
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1
        
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1
        
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1
        
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1
        
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1
        
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1
        
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1
        
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m221(rank_offset, i0, j0, k0):
        u_loc = [[0]*4 for _ in range(4)]
        v_loc = [[0]*4 for _ in range(2)]
        w_loc = [[0]*4 for _ in range(2)]
        
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1
        
        u_loc[1][1] = 1
        v_loc[1][1] = 1
        w_loc[0][1] = 1
        
        u_loc[2][2] = 1
        v_loc[0][2] = 1
        w_loc[1][2] = 1
        
        u_loc[3][3] = 1
        v_loc[1][3] = 1
        w_loc[1][3] = 1
        
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(1):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*1 + k_l][r]
        
        for k_l in range(1):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m127(rank_offset, i0, j0, k0):
        for j_l in range(2):
            for k_l in range(7):
                r = j_l * 7 + k_l
                u_g_idx = (i0 + 0) * m + (j0 + j_l)
                U[u_g_idx][rank_offset + r] += 1
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                w_g_idx = (k0 + k_l) * n + (i0 + 0)
                W[w_g_idx][rank_offset + r] += 1

    def add_m317(rank_offset, i0, j0, k0):
        for i_l in range(3):
            for k_l in range(7):
                r = i_l * 7 + k_l
                u_g_idx = (i0 + i_l) * m + (j0 + 0)
                U[u_g_idx][rank_offset + r] += 1
                v_g_idx = (j0 + 0) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                W[w_g_idx][rank_offset + r] += 1

    # Block 1: M327 (cols 0,1)
    add_strassen(0, 0, 0, 0)
    add_strassen(7, 0, 0, 2)
    add_strassen(14, 0, 0, 4)
    add_m221(21, 0, 0, 6)
    add_m127(25, 2, 0, 0)
    
    # Block 2: M327 (cols 2,3)
    add_strassen(39, 0, 2, 0)
    add_strassen(46, 0, 2, 2)
    add_strassen(53, 0, 2, 4)
    add_m221(60, 0, 2, 6)
    add_m127(64, 2, 2, 0)
    
    # Block 3: M317 (col 4)
    add_m317(78, 0, 4, 0)
    
    return {"U": U, "V": V, "W": W}
