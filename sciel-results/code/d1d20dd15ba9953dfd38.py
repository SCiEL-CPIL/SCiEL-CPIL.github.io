"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank-minimal exact decomposition for 2x4 * 4x7 matrix multiplication.
    # We use a block decomposition: split the 4x4 middle into 2x2 blocks.
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    # 
    # For each 2x2 * 2x7 multiplication, we can use Strassen for the 2x2*2x2 part
    # and schoolbook for the remaining columns.
    # 
    # Actually, let's think more carefully. We have 2x4 * 4x7.
    # Let's split as: A is 2x4, B is 4x7.
    # Split A into two 2x2 blocks: A = [A_left, A_right]
    # Split B into two 2x7 blocks: B = [B_top; B_bottom]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term is a 2x2 * 2x7 multiplication.
    # For 2x2 * 2x7: split the 2x7 into a 2x2 and a 2x5.
    # 2x2 * 2x2 can be done with Strassen (rank 7).
    # 2x2 * 2x5 can be done with schoolbook (rank 4*5=20) or further split.
    #
    # Total rank: 2 * (7 + 20) = 54. That's worse than schoolbook 56.
    #
    # Let me try another approach. Use the known result that the rank of 
    # [2,4,7] multiplication tensor is at most 50.
    #
    # Strategy: Split 4x4 into 2x2 blocks.
    # A (2x4) = [A1, A2] (each 2x2)
    # B (4x7) = [B1; B2] (each 2x7)
    # C = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x7:
    # Split 2x7 into 2x2 + 2x2 + 2x3
    # 2x2*2x2: Strassen rank 7
    # 2x2*2x2: Strassen rank 7  
    # 2x2*2x3: schoolbook rank 12
    # Total per 2x2*2x7: 7+7+12 = 26
    # Total: 2*26 = 52. Still not great.
    #
    # Let's try: 2x2 * 2x3 with Strassen-like.
    # Actually, for 2x2*2x3, we can use a rank-12 schoolbook or try to do better.
    #
    # Alternative: use the decomposition from the knowledge:
    # "Using a rank-50 decomposition by splitting the 2x2x3 blocks into 
    # 2x2x2 (Strassen, rank 7) and 2x2x1 (schoolbook, rank 4) components"
    #
    # Let's work with this. The idea seems to be:
    # Split the problem into smaller blocks and use Strassen where possible.
    #
    # Let me construct a rank-50 decomposition explicitly.
    #
    # A is 2x4, B is 4x7, C is 2x7.
    # 
    # Index A as A[i,j] for i in {0,1}, j in {0,1,2,3}
    # Index B as B[j,k] for j in {0,1,2,3}, k in {0,...,6}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Let's split j into {0,1} and {2,3}.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # Group 1: j in {0,1}: This is a 2x2 (rows i=0,1; cols j=0,1) times 2x7.
    # Group 2: j in {2,3}: This is a 2x2 (rows i=0,1; cols j=2,3) times 2x7.
    #
    # For each group, we have a 2x2 * 2x7 multiplication.
    #
    # For 2x2 * 2x7: Let X be 2x2, Y be 2x7.
    # Split Y into Y1 (2x2), Y2 (2x2), Y3 (2x3).
    # X*Y = X*Y1 + X*Y2 + X*Y3
    # X*Y1: 2x2*2x2, Strassen rank 7
    # X*Y2: 2x2*2x2, Strassen rank 7
    # X*Y3: 2x2*2x3, schoolbook rank 12
    # Total: 7+7+12 = 26 per group.
    # Two groups: 52.
    #
    # To get 50, we need to save 2 more ranks.
    #
    # Idea: For the 2x2*2x3 part, can we do better than 12?
    # 2x2*2x3: The tensor has rank at most 12 (schoolbook).
    # Can we do it in 11? Unlikely for such small dimensions.
    #
    # Alternative approach: Don't split uniformly.
    #
    # Let me try a different block structure.
    # Split the 4 columns of B (and rows of A) differently.
    #
    # Actually, let me just implement the rank-52 solution first and verify it works,
    # then see if I can find a rank-50.
    #
    # For now, let me implement a clean rank-52 solution using Strassen for 2x2*2x2.
    
    n, m, p = 2, 4, 7
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (28, rank)
    # W has shape (p*n, rank) = (14, rank)
    #
    # The decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the trace convention is W[k*n+i, r]. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # So the tensor is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And this should equal the standard matrix multiplication tensor.
    
    # Let me build the rank-52 solution.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U_cols.append([0]*8)
        V_cols.append([0]*28)
        W_cols.append([0]*14)
        U_cols[-1][u_idx] = u_val
        V_cols[-1][v_idx] = v_val
        W_cols[-1][w_idx] = w_val
    
    # Group 1: j in {0,1}, i in {0,1}
    # X = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # For each block of Y (2x7):
    
    # Y1: k in {0,1}
    # Y2: k in {2,3}
    # Y3: k in {4,5,6}
    
    # Strassen for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2_2x2(base_u, base_v, base_w, u_offsets, v_offsets, w_offsets):
        """Add Strassen terms for 2x2*2x2.
        u_offsets: list of 4 indices for A elements (row-major: [00,01,10,11])
        v_offsets: list of 4 indices for B elements (row-major: [00,01,10,11])
        w_offsets: list of 4 indices for C elements (row-major: [00,01,10,11])
        """
        a00, a01, a10, a11 = u_offsets
        b00, b01, b10, b11 = v_offsets
        c00, c01, c10, c11 = w_offsets
        
        # M1 = (A00+A11)*(B00+B11)
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a00] = 1; U_cols[-1][a11] = 1
        V_cols[-1][b00] = 1; V_cols[-1][b11] = 1
        W_cols[-1][c00] = 1; W_cols[-1][c11] = 1
        
        # M2 = (A10+A11)*B00
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a10] = 1; U_cols[-1][a11] = 1
        V_cols[-1][b00] = 1
        W_cols[-1][c10] = 1; W_cols[-1][c11] = -1
        
        # M3 = A00*(B01-B11)
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a00] = 1
        V_cols[-1][b01] = 1; V_cols[-1][b11] = -1
        W_cols[-1][c01] = 1; W_cols[-1][c11] = 1
        
        # M4 = A11*(B10-B00)
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a11] = 1
        V_cols[-1][b10] = 1; V_cols[-1][b00] = -1
        W_cols[-1][c00] = -1; W_cols[-1][c01] = 1
        
        # M5 = (A00+A01)*B11
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a00] = 1; U_cols[-1][a01] = 1
        V_cols[-1][b11] = 1
        W_cols[-1][c00] = 1; W_cols[-1][c10] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a10] = 1; U_cols[-1][a00] = -1
        V_cols[-1][b00] = 1; V_cols[-1][b01] = 1
        W_cols[-1][c11] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
        U_cols[-1][a01] = 1; U_cols[-1][a11] = -1
        V_cols[-1][b10] = 1; V_cols[-1][b11] = 1
        W_cols[-1][c00] = 1; W_cols[-1][c10] = -1
    
    def schoolbook_2x2_2x3(base_u, base_v, base_w, u_offsets, v_offsets, w_offsets):
        """Add schoolbook terms for 2x2*2x3.
        u_offsets: 4 indices for A (2x2)
        v_offsets: 6 indices for B (2x3), row-major
        w_offsets: 6 indices for C (2x3), row-major
        """
        a00, a01, a10, a11 = u_offsets
        b00, b01, b02, b10, b11, b12 = v_offsets
        c00, c01, c02, c10, c11, c12 = w_offsets
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_cols.append([0]*8); V_cols.append([0]*28); W_cols.append([0]*14)
                    U_cols[-1][u_offsets[i*2+j]] = 1
                    V_cols[-1][v_offsets[j*3+k]] = 1
                    W_cols[-1][w_offsets[i*3+k]] = 1
    
    # Group 1: j in {0,1}
    # A elements: A[i,0], A[i,1] for i in {0,1}
    # U indices: i*m + j = i*4 + j
    # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 4; i=1,j=1 -> 5
    u_g1 = [0, 1, 4, 5]  # a00, a01, a10, a11
    
    # Y1: k in {0,1}, B[j,k] for j in {0,1}
    # V indices: j*p + k
    # j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 7; j=1,k=1 -> 8
    v_g1_y1 = [0, 1, 7, 8]  # b00, b01, b10, b11
    
    # W indices: k*n + i
    # k=0,i=0 -> 0; k=1,i=0 -> 2; k=0,i=1 -> 1; k=1,i=1 -> 3
    w_g1_y1 = [0, 2, 1, 3]  # c00, c01, c10, c11
    
    strassen_2x2_2x2(0, 0, 0, u_g1, v_g1_y1, w_g1_y1)
    
    # Y2: k in {2,3}
    # j=0,k=2 -> 2; j=0,k=3 -> 3; j=1,k=2 -> 9; j=1,k=3 -> 10
    v_g1_y2 = [2, 3, 9, 10]
    # k=2,i=0 -> 4; k=3,i=0 -> 6; k=2,i=1 -> 5; k=3,i=1 -> 7
    w_g1_y2 = [4, 6, 5, 7]
    
    strassen_2x2_2x2(0, 0, 0, u_g1, v_g1_y2, w_g1_y2)
    
    # Y3: k in {4,5,6}
    # j=0,k=4 -> 4; j=0,k=5 -> 5; j=0,k=6 -> 6
    # j=1,k=4 -> 11; j=1,k=5 -> 12; j=1,k=6 -> 13
    v_g1_y3 = [4, 5, 6, 11, 12, 13]
    # k=4,i=0 -> 8; k=5,i=0 -> 10; k=6,i=0 -> 12
    # k=4,i=1 -> 9; k=5,i=1 -> 11; k=6,i=1 -> 13
    w_g1_y3 = [8, 10, 12, 9, 11, 13]
    
    schoolbook_2x2_2x3(0, 0, 0, u_g1, v_g1_y3, w_g1_y3)
    
    # Group 2: j in {2,3}
    # A elements: A[i,2], A[i,3] for i in {0,1}
    # i=0,j=2 -> 2; i=0,j=3 -> 3; i=1,j=2 -> 6; i=1,j=3 -> 7
    u_g2 = [2, 3, 6, 7]
    
    # Y1: k in {0,1}
    v_g2_y1 = [0, 1, 7, 8]  # same as before
    w_g2_y1 = [0, 2, 1, 3]  # same as before
    
    strassen_2x2_2x2(0, 0, 0, u_g2, v_g2_y1, w_g2_y1)
    
    # Y2: k in {2,3}
    v_g2_y2 = [2, 3, 9, 10]
    w_g2_y2 = [4, 6, 5, 7]
    
    strassen_2x2_2x2(0, 0, 0, u_g2, v_g2_y2, w_g2_y2)
    
    # Y3: k in {4,5,6}
    v_g2_y3 = [4, 5, 6, 11, 12, 13]
    w_g2_y3 = [8, 10, 12, 9, 11, 13]
    
    schoolbook_2x2_2x3(0, 0, 0, u_g2, v_g2_y3, w_g2_y3)
    
    rank = len(U_cols)
    
    # Build final matrices
    U = [U_cols[c] for c in range(rank)]
    V = [V_cols[c] for c in range(rank)]
    W = [W_cols[c] for c in range(rank)]
    
    # Transpose to get shape (n*m, rank), (m*p, rank), (p*n, rank)
    U_final = [[U[c][i] for c in range(rank)] for i in range(n*m)]
    V_final = [[V[c][i] for c in range(rank)] for i in range(m*p)]
    W_final = [[W[c][i] for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
