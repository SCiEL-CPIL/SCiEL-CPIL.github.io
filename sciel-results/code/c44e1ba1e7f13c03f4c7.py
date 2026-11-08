"""Matrix multiplication 4x4x8 with Strassen-based decomposition."""
DIMENSIONS = [4, 4, 8]

import time

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns U2 (4x7), V2 (4x7), W2 (4x7) with entries in {0, 1, -1}.
    
    Indexing: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    """
    rank = 7
    U2 = [[0]*rank for _ in range(4)]
    V2 = [[0]*rank for _ in range(4)]
    W2 = [[0]*rank for _ in range(4)]
    
    # M1 = A00*(B01 - B11)
    U2[0][0] = 1  # A00
    V2[1][0] = 1  # B01
    V2[3][0] = -1  # -B11
    W2[1][0] = 1  # C01
    W2[3][0] = 1  # C11
    
    # M2 = A10*(B00 + B01)
    U2[2][1] = 1  # A10
    V2[0][1] = 1  # B00
    V2[1][1] = 1  # B01
    W2[0][1] = 1  # C10
    W2[2][1] = 1  # C11
    
    # M3 = (A00 + A10)*B10
    U2[0][2] = 1  # A00
    U2[2][2] = 1  # A10
    V2[2][2] = 1  # B10
    W2[0][2] = 1  # C00
    W2[2][2] = 1  # C10
    
    # M4 = (A01 - A11)*B11
    U2[1][3] = 1  # A01
    U2[3][3] = -1  # -A11
    V2[3][3] = 1  # B11
    W2[1][3] = 1  # C01
    W2[3][3] = 1  # C11
    
    # M5 = (A00 - A11)*(B10 + B11)
    U2[0][4] = 1  # A00
    U2[3][4] = -1  # -A11
    V2[2][4] = 1  # B10
    V2[3][4] = 1  # B11
    W2[0][4] = 1  # C00
    W2[1][4] = 1  # C01
    W2[2][4] = 1  # C10
    W2[3][4] = 1  # C11
    
    # M6 = (A01 + A11)*(B00 - B10)
    U2[1][5] = 1  # A01
    U2[3][5] = 1  # A11
    V2[0][5] = 1  # B00
    V2[2][5] = -1  # -B10
    W2[0][5] = 1  # C00
    W2[1][5] = 1  # C01
    W2[2][5] = 1  # C10
    W2[3][5] = 1  # C11
    
    # M7 = (A00 + A01)*B11
    # Wait, let me recheck. Standard Strassen:
    # M1 = A00*(B01 - B11)
    # M2 = (A10 + A11)*B00
    # M3 = (A00 + A01)*B11
    # M4 = A11*(B10 - B00)
    # M5 = (A00 + A10)*(B00 + B01)
    # M6 = (A01 - A10)*(B10 + B11)
    # M7 = (A00 - A11)*(B01 + B11)
    # C00 = M5 + M7 - M2 + M4
    # C01 = M1 + M3
    # C10 = M2 + M6
    # C11 = M3 + M5 - M1 - M7
    #
    # Let me redo with the standard formulation.
    
    U2 = [[0]*rank for _ in range(4)]
    V2 = [[0]*rank for _ in range(4)]
    W2 = [[0]*rank for _ in range(4)]
    
    # M1 = A00*(B01 - B11)
    U2[0][0] = 1
    V2[1][0] = 1
    V2[3][0] = -1
    W2[1][0] = 1
    W2[3][0] = 1
    
    # M2 = (A10 + A11)*B00
    U2[2][1] = 1
    U2[3][1] = 1
    V2[0][1] = 1
    W2[0][1] = -1
    W2[2][1] = 1
    
    # M3 = (A00 + A01)*B11
    U2[0][2] = 1
    U2[1][2] = 1
    V2[3][2] = 1
    W2[1][2] = 1
    W2[3][2] = 1
    
    # M4 = A11*(B10 - B00)
    U2[3][3] = 1
    V2[2][3] = 1
    V2[0][3] = -1
    W2[0][3] = 1
    W2[2][3] = 1
    
    # M5 = (A00 + A10)*(B00 + B01)
    U2[0][4] = 1
    U2[2][4] = 1
    V2[0][4] = 1
    V2[1][4] = 1
    W2[0][4] = 1
    W2[2][4] = 1
    
    # M6 = (A01 - A10)*(B10 + B11)
    U2[1][5] = 1
    U2[2][5] = -1
    V2[2][5] = 1
    V2[3][5] = 1
    W2[2][5] = 1
    W2[3][5] = 1
    
    # M7 = (A00 - A11)*(B01 + B11)
    U2[0][6] = 1
    U2[3][6] = -1
    V2[1][6] = 1
    V2[3][6] = 1
    W2[0][6] = 1
    W2[1][6] = 1
    W2[2][6] = 1
    W2[3][6] = 1
    
    return U2, V2, W2

def _strassen_4x4():
    """
    Strassen decomposition for 4x4 x 4x4 -> 4x4 using recursive Strassen.
    Returns U4 (16x49), V4 (16x49), W4 (16x49).
    
    Indexing: U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    """
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank4 = rank2 * rank2  # 49
    
    U4 = [[0]*rank4 for _ in range(16)]
    V4 = [[0]*rank4 for _ in range(16)]
    W4 = [[0]*rank4 for _ in range(16)]
    
    # For each outer Strassen term (a,b) and inner Strassen term (i',j'),k'
    # Global indices: i = 2*ai + i', j = 2*bj + j', k = 2*ck + k'
    # But wait, for 4x4, we partition into 2x2 blocks of 2x2 matrices.
    # Block (a,b) where a,b in {0,1}, each block is 2x2.
    # i = 2*a + i', j = 2*b + j', k = 2*c + k'
    
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for ci in range(2):
                    for cj in range(2):
                        # Outer level: block indices a, b, c
                        # Inner level: scalar indices ci, cj, ck
                        for outer_c in range(rank2):
                            # Get the outer factor values
                            # U2 is indexed by [i*2+j, c] for 2x2
                            # For block (a,b): i=a, j=b in the 2x2 block space
                            U_val = U2[a*2+b][outer_c]
                            V_val = V2[b*2+c][outer_c]
                            W_val = W2[c*2+a][outer_c]
                            
                            for inner_c in range(rank2):
                                # Inner level: 2x2 multiplication within the block
                                # For block (b,c) on V side, (a,b) on U side, (c,a) on W side
                                # But actually the inner indices are local to the block
                                U_inner_val = U2[ci*2+cj][inner_c]
                                V_inner_val = V2[cj*2+ck][inner_c]  # Hmm, need to be careful
                                W_inner_val = W2[ck*2+ci][inner_c]
                                
                                # Wait, I think I'm overcomplicating this.
                                # The 4x4x4 tensor is a Kronecker product structure.
                                # T_4[i,j,k] = sum_{a,b,c} T_2[a,b,c] * T_2[i-a, j-b, k-c]
                                # where the inner indices are local to each 2x2 block.
                                
                                # Actually, let me think of it differently.
                                # A 4x4 matrix can be written as 2x2 blocks of 2x2 matrices.
                                # A = [[A00, A01], [A10, A11]] where each Ajk is 2x2
                                # B = [[B00, B01], [B10, B11]] where each Bjk is 2x2
                                # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
                                #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
                                # Each 2x2 block multiplication can be done with Strassen rank 7.
                                # So total rank = 4 * 7 = 28? No, that's not right either.
                                
                                # Actually, the standard recursive Strassen gives rank 7^2 = 49 for 4x4.
                                # Let me just construct it properly.
                                pass
    # This approach is getting complicated. Let me use a different construction.
    # The 4x4x4 tensor can be decomposed as a Kronecker product of two 2x2x2 tensors.
    # T_4 = T_2 ⊗ T_2 where the Kronecker product is applied appropriately.
    
    # For the Kronecker product structure:
    # If T2 has factors U2 (4x7), V2 (4x7), W2 (4x7),
    # then T4 = T2 ⊗ T2 has rank 7*7 = 49
    # U4[i*4+j, c1*7+c2] = U2[i*2+i2, c1] * U2[j*2+j2, c2] ... this isn't quite right.
    
    # Let me just do the explicit construction by verifying each entry.
    # Actually, the simplest approach: use the fact that rank(4x4x4) <= 49 via Strassen.
    # Let me construct it by iterating over all (outer_c, inner_c) pairs.
    
    U4 = [[0]*rank4 for _ in range(16)]
    V4 = [[0]*rank4 for _ in range(16)]
    W4 = [[0]*rank4 for _ in range(16)]
    
    for outer_c in range(rank2):
        for inner_c in range(rank2):
            col = outer_c * rank2 + inner_c
            # For each combination of outer and inner Strassen terms
            for a in range(2):
                for b in range(2):
                    U_ou = U2[a*2+b][outer_c]
                    if U_ou == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            U_in = U2[i2*2+j2][inner_c]
                            if U_in == 0:
                                continue
                            i = 2*a + i2
                            j = 2*b + j2
                            U4[i*4+j][col] += U_ou * U_in
                            
            for b in range(2):
                for c in range(2):
                    V_ou = V2[b*2+c][outer_c]
                    if V_ou == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            V_in = V2[j2*2+k2][inner_c]
                            if V_in == 0:
                                continue
                            j = 2*b + j2
                            k = 2*c + k2
                            V4[j*4+k][col] += V_ou * V_in
                            
            for c in range(2):
                for a in range(2):
                    W_ou = W2[c*2+a][outer_c]
                    if W_ou == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            W_in = W2[k2*2+i2][inner_c]
                            if W_in == 0:
                                continue
                            k = 2*c + k2
                            i = 2*a + i2
                            W4[k*4+i][col] += W_ou * W_in
                            
    return U4, V4, W4

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 -> 4x8.
    
    Uses recursive Strassen: rank 49 for 4x4x4, duplicated for the two output blocks.
    Total rank: 98.
    
    Returns: {'U': (16, rank), 'V': (32, rank), 'W': (32, rank)}
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    U4, V4, W4 = _strassen_4x4()
    rank4 = 49
    total_rank = rank4 * 2  # 98
    
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 98
    V = [[0]*total_rank for _ in range(m*p)]  # 32 x 98
    W = [[0]*total_rank for _ in range(p*n)]  # 32 x 98
    
    # Block 1: k in {0,1,2,3}
    for i in range(n):
        for j in range(m):
            for c in range(rank4):
                U[i*m+j][c] = U4[i*m+j][c]
    
    for j in range(m):
        for k in range(4):
            for c in range(rank4):
                V[j*p+k][c] = V4[j*4+k][c]
    
    for k in range(4):
        for i in range(n):
            for c in range(rank4):
                W[k*n+i][c] = W4[k*n+i][c]
    
    # Block 2: k in {4,5,6,7}
    offset = rank4
    for i in range(n):
        for j in range(m):
            for c in range(rank4):
                U[i*m+j][offset+c] = U4[i*m+j][c]
    
    for j in range(m):
        for k in range(4, 8):
            for c in range(rank4):
                V[j*p+k][offset+c] = V4[j*4+(k-4)][c]
    
    for k in range(4, 8):
        for i in range(n):
            for c in range(rank4):
                W[k*n+i][offset+c] = W4[(k-4)*n+i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
