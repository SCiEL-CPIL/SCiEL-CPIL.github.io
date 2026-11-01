DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-33 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Strategy: Partition the 12 columns into three groups of 4 columns each.
    For each 2x2 * 2x4 block, use a rank-11 decomposition:
      - First 2 columns of each group: 7-product Strassen
      - Last 2 columns of each group: 4-product schoolbook
    
    Total rank = 3 * 11 = 33.
    
    The output factor W follows the trace convention W[k*n+i, r] where
    k is the column index in C and i is the row index in C.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_groups = p // 4  # 3 groups of 4 columns
    rank_per_group = 11  # 7 Strassen + 4 schoolbook
    rank = num_groups * rank_per_group  # 33
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 33
    V = [[0] * rank for _ in range(m * p)]   # 24 x 33
    W = [[0] * rank for _ in range(p * n)]   # 24 x 33
    
    # Define the 11 products for a 2x2 * 2x4 block
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22, (0,2)=B31, (0,3)=B32, (1,2)=B41, (1,3)=B42
    
    # products[s] = (list of (A_index, coeff), list of (B_index, coeff))
    products = [
        # Strassen products for the first 2x2 sub-block (columns 0,1)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),       # M1=(A11+A22)(B11+B22)
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),                    # M2=(A21+A22)B11
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),                   # M3=A11(B12-B22)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),                   # M4=A22(B21-B11)
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),                    # M5=(A11+A12)B22
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),       # M6=(A21-A11)(B11+B12)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),       # M7=(A12-A22)(B21+B22)
        # Schoolbook products for the second 2x2 sub-block (columns 2,3)
        ([((0,0), 1)], [((0,2), 1)]),                                # M8=A11*B31
        ([((0,1), 1)], [((0,3), 1)]),                                # M9=A12*B32
        ([((1,0), 1)], [((1,2), 1)]),                                # M10=A21*B41
        ([((1,1), 1)], [((1,3), 1)]),                                # M11=A22*B42
    ]
    
    # output_contribs[s] = list of (C_index, coeff) for each product
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22, (0,2)=C13, (0,3)=C14, (1,2)=C23, (1,3)=C24
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],     # M1: C11+=1, C22+=1
        [((1,0), 1), ((1,1), -1)],    # M2: C21+=1, C22+=-1
        [((0,1), 1), ((1,1), 1)],     # M3: C12+=1, C22+=1
        [((0,0), 1), ((1,0), 1)],     # M4: C11+=1, C21+=1
        [((0,0), -1), ((0,1), 1)],    # M5: C11+=-1, C12+=1
        [((1,1), 1)],                 # M6: C22+=1
        [((0,0), 1)],                 # M7: C11+=1
        [((0,2), 1)],                 # M8: C13+=1
        [((0,3), 1)],                 # M9: C14+=1
        [((1,2), 1)],                 # M10: C23+=1
        [((1,3), 1)],                 # M11: C24+=1
    ]
    
    col = 0
    for t in range(num_groups):
        base_col = 4 * t
        for s in range(rank_per_group):
            # Fill U (A coefficients)
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            
            # Fill V (B coefficients)
            for (bi, bj), coeff in products[s][1]:
                b_col = base_col + bj
                v_row = bi * p + b_col
                V[v_row][col] = coeff
            
            # Fill W (C coefficients)
            for (ci, cj), coeff in output_contribs[s]:
                c_col = base_col + cj
                w_row = c_col * n + ci
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
