import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a rank-97 tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
    - Split the 4x4 A and 4x7 B into 2x2 blocks.
    - The B matrix has 4 columns of width 2, plus 1 column of width 1.
    - For the first 4 columns (2x2 blocks of B), use Strassen's 7-term scheme for 2x2x2.
      This yields 4 (column blocks) * 7 (Strassen terms) = 28 terms.
      Each term involves a 2x2 multiplication, so we need to decompose the 2x2x2 tensor.
      Wait, the tensor decomposition for 2x2x2 has rank 7.
      So for each of the 4 column-blocks of B, we use the 7-term Strassen decomposition for the 2x2x2 part.
      This gives 4 * 7 = 28 terms for the 4x4x4 part (first 4 columns of C).
    - For the last column (width 1) of B, use schoolbook multiplication for 4x4 * 4x1.
      This requires 4*4*1 = 16 terms.
    
    Total rank = 28 + 16 = 44? No, that's not right.
    
    Let's re-evaluate.
    
    The tensor is T_{i,j,k} = sum_l A_{i,l} B_{l,k}.
    We want U_{i,l,t}, V_{l,k,t}, W_{i,k,t} such that T_{i,j,k} = sum_t U_{i,j,t} V_{j,k,t} W_{i,k,t}.
    
    If we use Strassen for 2x2x2, the rank is 7.
    For 4x4x4, we can view it as 2x2 blocks of 2x2 matrices.
    The multiplication of two 2x2 block matrices (each block is 2x2) requires 7 scalar multiplications of 2x2 matrices.
    Each scalar multiplication of 2x2 matrices has rank 7.
    So the total rank is 7 * 7 = 49 for 4x4x4.
    
    For the 4x4x1 part (last column), we can use schoolbook: 4*4*1 = 16 terms.
    But wait, the B matrix is 4x7. We can split it into four 4x2 blocks and one 4x1 block.
    The first four blocks are 4x2. We can treat each 4x2 block as two 2x2 blocks? No, that's not how it works.
    
    Let's think differently.
    The B matrix is 4x7. Let's split the 7 columns into four groups of 2 and one group of 1.
    For each group of 2 columns, we have a 4x2 matrix.
    The multiplication A (4x4) * B_block (4x2) gives a 4x2 result.
    We can decompose the 4x4 * 4x2 multiplication.
    
    Alternatively, we can use the known rank-49 decomposition for 4x4x4 and rank-16 for 4x4x1.
    Total rank = 49 + 16 = 65.
    
    Let's verify the rank-49 decomposition for 4x4x4.
    It is constructed by applying Strassen's 7-term scheme to the 2x2 blocks.
    Each of the 7 terms involves a multiplication of two 2x2 matrices.
    Each 2x2x2 multiplication has rank 7.
    So total rank is 7*7 = 49.
    
    For the 4x4x1 part, schoolbook gives rank 16.
    
    Total rank = 49 + 16 = 65.
    
    Let's implement this.
    
    Step 1: Generate the 49 terms for 4x4x4.
    Step 2: Generate the 16 terms for 4x4x1.
    
    For Step 1:
    We have 7 Strassen terms for the 2x2 block multiplication.
    For each term, we have 7 sub-terms for the 2x2x2 multiplication.
    So 49 terms total.
    
    Let's define the Strassen 2x2x2 decomposition as U, V, W matrices of size 4x7.
    Then for the 4x4x4 case, we combine them.
    
    Actually, let's just use the schoolbook for simplicity and correctness, then optimize.
    Schoolbook rank = 4*4*7 = 112.
    
    The task asks for a rank-minimal exact tensor decomposition.
    The known minimum rank for 4x4x4 is 49 (Strassen).
    For 4x4x1, it is 16 (schoolbook is optimal for 1-wide).
    
    So the target rank is 49 + 16 = 65.
    
    Let's construct the 49 terms for 4x4x4.
    
    We'll use the recursive Strassen construction.
    
    def strassen_2x2():
        Returns U, V, W for 2x2x2.
        U, V, W are 4x7 matrices.
    
    def strassen_4x4():
        Returns U, V, W for 4x4x4.
        U, V, W are 16x49 matrices.
    
    We'll implement strassen_4x4 by combining strassen_2x2.
    
    The 7 Strassen terms for 2x2x2 are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    For 4x4x4, we treat A and B as 2x2 block matrices, where each block is 2x2.
    So we have 7 block multiplications, each of which is a 2x2x2 multiplication.
    
    For each block multiplication Mi, we use the 2x2x2 Strassen decomposition.
    
    Let's define the mapping.
    
    A is 4x4. Blocks: A00, A01, A10, A11.
    B is 4x4. Blocks: B00, B01, B10, B11.
    C is 4x4. Blocks: C00, C01, C10, C11.
    
    The 7 block terms are:
    T1 = (A00+A11)(B00+B11)
    T2 = (A10+A11)B00
    T3 = A00(B01-B11)
    T4 = A11(B10-B00)
    T5 = (A00+A01)B11
    T6 = (A10-A00)(B00+B01)
    T7 = (A01-A11)(B10+B11)
    
    Each Ti is a 2x2 matrix.
    C00 = T1 + T4 + T7
    C01 = T3 + T5
    C10 = T2 + T4
    C11 = T1 - T5 + T6
    
    Now, for each Ti, we decompose the 2x2 multiplication into 7 scalar multiplications.
    
    Let's denote the 2x2x2 Strassen decomposition as U2, V2, W2.
    U2[i][t] = coefficient for A element i in term t.
    V2[j][t] = coefficient for B element j in term t.
    W2[k][t] = coefficient for C element k in term t.
    
    For the 4x4x4 case, we have 7 block terms, each with 7 sub-terms.
    So 49 terms total.
    
    Let's index the 49 terms as (block_term, sub_term).
    
    For a given block_term b and sub_term s:
    The A part is a linear combination of the 2x2 blocks involved in Tb.
    The B part is a linear combination of the 2x2 blocks involved in Tb.
    The C part is the contribution of Tb to the C blocks, multiplied by the sub-term's C coefficient.
    
    Let's define the A and B block coefficients for each Tb.
    
    Tb1: A00:1, A11:1; B00:1, B11:1
    Tb2: A10:1, A11:1; B00:1
    Tb3: A00:1; B01:1, B11:-1
    Tb4: A11:1; B10:1, B00:-1
    Tb5: A00:1, A01:1; B11:1
    Tb6: A10:1, A00:-1; B00:1, B01:1
    Tb7: A01:1, A11:-1; B10:1, B11:1
    
    For each Tb, we have a 2x2 matrix X and Y, where X is the A part and Y is the B part.
    X is a 2x2 matrix, Y is a 2x2 matrix.
    The product Z = X * Y is a 2x2 matrix.
    
    We decompose Z = sum_s U2[s] * X_s * Y_s * W2[s].
    
    Wait, the 2x2x2 Strassen decomposition is:
    Z = sum_s (sum_i U2[i][s] X_i) * (sum_j V2[j][s] Y_j) * W2[s]
    
    No, that's not right. The decomposition is:
    Z_{kl} = sum_s U2[k][s] * V2[l][s] * W2[s]
    
    Where U2, V2, W2 are 4x7 matrices.
    
    Actually, the standard form is:
    T_{i,j,k} = sum_t U_{i,t} V_{j,t} W_{k,t}
    
    For 2x2x2, i,j,k in {0,1,2,3} (flattened 2x2 indices).
    
    So for the 4x4x4 case, we have 16 A elements, 16 B elements, 16 C elements.
    
    Let's map the 2x2 block indices to 4x4 indices.
    
    A block (br, bc) has elements at (2*br + r, 2*bc + c) for r,c in {0,1}.
    
    For a given block term b and sub term s:
    The A part is a linear combination of the blocks in Tb.
    Let a_coeff[br, bc] be the coefficient for block (br, bc) in Tb.
    Then for each element (r, c) in the 2x2 block, the coefficient in U is a_coeff[br, bc] * U2[r*2+c][s].
    
    Similarly for B and C.
    
    For C, the block term Tb contributes to specific C blocks with specific signs.
    Let c_coeff[br, bc] be the coefficient for C block (br, bc) from Tb.
    Then for each element (r, c) in the 2x2 block, the coefficient in W is c_coeff[br, bc] * W2[r*2+c][s].
    
    Let's define the c_coeff for each Tb.
    
    Tb1: C00:1, C11:1
    Tb2: C10:1
    Tb3: C01:1
    Tb4: C00:1, C10:1
    Tb5: C01:1, C11:-1
    Tb6: C11:1
    Tb7: C00:1, C11:-1
    
    Now we can construct the 49 terms.
    
    For the 4x4x1 part, we use schoolbook: 16 terms.
    
    Total rank = 49 + 16 = 65.
    
    Let's implement this.
    
    """
    n, m, p = 4, 4, 7
    total_rank = 65
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1; W2[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    # Block coefficients for the 7 Strassen terms
    # a_coeff[b][br][bc], b_coeff[b][br][bc], c_coeff[b][br][bc]
    a_coeff = [
        { (0,0):1, (1,1):1 },  # T1
        { (1,0):1, (1,1):1 },  # T2
        { (0,0):1 },            # T3
        { (1,1):1 },            # T4
        { (0,0):1, (0,1):1 },  # T5
        { (1,0):1, (0,0):-1 }, # T6
        { (0,1):1, (1,1):-1 }, # T7
    ]
    b_coeff = [
        { (0,0):1, (1,1):1 },  # T1
        { (0,0):1 },            # T2
        { (0,1):1, (1,1):-1 }, # T3
        { (1,0):1, (0,0):-1 }, # T4
        { (1,1):1 },            # T5
        { (0,0):1, (0,1):1 },  # T6
        { (1,0):1, (1,1):1 },  # T7
    ]
    c_coeff = [
        { (0,0):1, (1,1):1 },  # T1
        { (1,0):1 },            # T2
        { (0,1):1 },            # T3
        { (0,0):1, (1,0):1 },  # T4
        { (0,1):1, (1,1):-1 }, # T5
        { (1,1):1 },            # T6
        { (0,0):1, (1,1):-1 }, # T7
    ]
    
    # Construct the 49 terms for 4x4x4
    for b in range(7):
        for s in range(7):
            t = b * 7 + s
            # U part
            for br in range(2):
                for bc in range(2):
                    if (br, bc) not in a_coeff[b]:
                        continue
                    coeff_a = a_coeff[b][(br, bc)]
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            global_idx = (br * 2 + r) * m + (bc * 2 + c)
                            U[global_idx][t] += U2[local_idx][s] * coeff_a
            # V part
            for br in range(2):
                for bc in range(2):
                    if (br, bc) not in b_coeff[b]:
                        continue
                    coeff_b = b_coeff[b][(br, bc)]
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            global_idx = (br * 2 + r) * p + (bc * 2 + c)
                            V[global_idx][t] += V2[local_idx][s] * coeff_b
            # W part
            for br in range(2):
                for bc in range(2):
                    if (br, bc) not in c_coeff[b]:
                        continue
                    coeff_c = c_coeff[b][(br, bc)]
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            global_idx = (bc * 2 + c) * n + (br * 2 + r)
                            W[global_idx][t] += W2[local_idx][s] * coeff_c
    
    # Construct the 16 terms for 4x4x1 (last column of B)
    base = 49
    for i in range(n):
        for j in range(m):
            for k in range(1):  # Only the last column (k=6)
                t = base + i * m + j
                u_idx = i * m + j
                v_idx = j * p + (p - 1 + k)  # k=0 corresponds to column 6
                w_idx = (p - 1 + k) * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
