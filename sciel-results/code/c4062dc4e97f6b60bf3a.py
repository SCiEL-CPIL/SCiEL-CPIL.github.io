import numpy as np
import time

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Constructs a rank-42 tensor decomposition for 2x2 * 2x12 matrix multiplication.
    This is achieved by partitioning the 12 columns of the second matrix into 6 blocks
    of 2 columns each, and applying the rank-7 Strassen decomposition to each block.
    Since the column indices of the blocks are disjoint, the ranks add up: 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = 150 - 5  # Safety margin
    
    n, m, p = DIMENSIONS
    num_pairs = p // 2  # 6 pairs of columns
    
    rank = num_pairs * 7
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen 7 products for 2x2 * 2x2
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements within block: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    products = [
        # M1=(A11+A22)(B11+B22)
        ( [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)] ),
        # M2=(A21+A22)B11
        ( [((1,0), 1), ((1,1), 1)], [((0,0), 1)] ),
        # M3=A11(B12-B22)
        ( [((0,0), 1)], [((0,1), 1), ((1,1), -1)] ),
        # M4=A22(B21-B11)
        ( [((1,1), 1)], [((1,0), 1), ((0,0), -1)] ),
        # M5=(A11+A12)B22
        ( [((0,0), 1), ((0,1), 1)], [((1,1), 1)] ),
        # M6=(A21-A11)(B11+B12)
        ( [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)] ),
        # M7=(A12-A22)(B21+B22)
        ( [((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)] ),
    ]
    
    output_contribs = [
        # M1: C11+=1, C22+=1
        [((0,0), 1), ((1,1), 1)],
        # M2: C21+=1, C22+=-1
        [((1,0), 1), ((1,1), -1)],
        # M3: C12+=1, C22+=1
        [((0,1), 1), ((1,1), 1)],
        # M4: C11+=1, C21+=1
        [((0,0), 1), ((1,0), 1)],
        # M5: C11+=-1, C12+=1
        [((0,0), -1), ((0,1), 1)],
        # M6: C22+=1
        [((1,1), 1)],
        # M7: C11+=1
        [((0,0), 1)],
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            if time.time() > deadline:
                # Return partial or best known if timeout
                # But this loop is very fast, so unlikely to hit
                pass
            
            # Fill U
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            
            # Fill V
            for (bj, bk_rel), coeff in products[s][1]:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row][col] = coeff
            
            # Fill W
            for (ci, ck_rel), coeff in output_contribs[s]:
                c_col = base_col + ck_rel
                # W index: k*n + i
                # ci is row i, ck_rel is col j
                w_row = c_col * n + ci
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
