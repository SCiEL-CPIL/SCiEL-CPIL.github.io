import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors of length 4.
    """
    # Indices: 00, 01, 10, 11
    # A: 0,1,2,3
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We need to expand these into scalar multiplications.
    # Since we are at the base case of the recursion for n=2, 
    # we actually need the rank-1 tensors for the 2x2x2 tensor.
    # But wait, the rank of 2x2x2 is 7.
    # The 7 terms P1..P7 are the rank-1 tensors?
    # No, P1 is a matrix product. 
    # The tensor M_2,2,2 is the sum of 7 rank-1 tensors.
    # Each rank-1 tensor corresponds to a scalar multiplication 
    # of a linear form in A and a linear form in B, 
    # contributing to a linear form in C.
    
    # However, the standard Strassen algorithm computes 7 matrix products.
    # If we consider the 2x2x2 tensor as an object, its rank is 7.
    # The decomposition is:
    # T = sum_{k=1}^7 u_k v_k w_k
    # where u_k, v_k, w_k are vectors of length 4.
    
    # Let's define them.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = []
    V = []
    W = []
    
    # P1: (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11?
    # No, P1 is a 2x2 matrix.
    # P1 = [[A00+A11, A00+A11], [A00+A11, A00+A11]] * [[B00+B11, B00+B11], [B00+B11, B00+B11]]?
    # No.
    # A_block = [[A00, A01], [A10, A11]]
    # B_block = [[B00, B01], [B10, B11]]
    # P1 = (A00+A11)(B00+B11) is a scalar?
    # No, in Strassen, A00 etc are 1x1 matrices (scalars) if n=2.
    # So P1 is a scalar.
    # P1 = (A00+A11)(B00+B11).
    # This scalar contributes to C00, C01, C10, C11.
    # C00 = P1 + P4 - P3 - P7
    # C01 = P5 + P7
    # C10 = P6 - P4
    # C11 = P2 + P3 - P6 - P7
    
    # So, each Pk is a scalar multiplication of two linear forms.
    # The linear form for A in Pk is a vector u_k.
    # The linear form for B in Pk is a vector v_k.
    # The contribution to C is w_k, which is a vector indicating which C entries get this scalar.
    
    # P1: A: 1,0,0,1. B: 1,0,0,1. C: 1,0,0,1 (coeff 1 in C00, 0 in others? No)
    # C00 gets +P1. C01 gets 0. C10 gets 0. C11 gets 0.
    # Wait, C00 = P1 + ...
    # So w_1 = [1, 0, 0, 0].
    
    # P2: (A10+A11)B00.
    # A: 0,0,1,1. B: 1,0,0,0.
    # C11 gets +P2.
    # w_2 = [0, 0, 0, 1].
    
    # P3: A00(B01-B11).
    # A: 1,0,0,0. B: 0,1,0,-1.
    # C00 gets -P3. C11 gets +P3.
    # w_3 = [-1, 0, 0, 1].
    
    # P4: A11(B10-B00).
    # A: 0,0,0,1. B: -1,0,1,0.
    # C00 gets +P4. C10 gets -P4.
    # w_4 = [1, 0, -1, 0].
    
    # P5: (A00+A01)B11.
    # A: 1,1,0,0. B: 0,0,0,1.
    # C01 gets +P5.
    # w_5 = [0, 1, 0, 0].
    
    # P6: (A10-A00)(B00+B01).
    # A: -1,0,1,0. B: 1,1,0,0.
    # C10 gets +P6. C11 gets -P6.
    # w_6 = [0, 0, 1, -1].
    
    # P7: (A01-A11)(B10+B11).
    # A: 0,1,0,-1. B: 0,0,1,1.
    # C00 gets -P7. C01 gets +P7. C11 gets -P7.
    # w_7 = [-1, 1, 0, -1].
    
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 1],
        [1, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, -1]
    ]
    return U, V, W

def _combine_strassen(n):
    """
    Generates Strassen decomposition for n x n x n where n is a power of 2.
    Returns U, V, W as lists of lists.
    """
    if n == 2:
        return _strassen_2x2x2()
    
    m = n // 2
    U_sub, V_sub, W_sub = _combine_strassen(m)
    rank_sub = len(U_sub)
    
    U = []
    V = []
    W = []
    
    # 7 Strassen products
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Indices in n x n matrix:
    # Block 11: 0..m-1, 0..m-1
    # Block 12: 0..m-1, m..2m-1
    # Block 21: m..2m-1, 0..m-1
    # Block 22: m..2m-1, m..2m-1
    
    # We need to map the sub-decomposition (size m) to the full size n.
    # For each of the 7 products, we have rank_sub terms.
    # Total rank = 7 * rank_sub.
    
    # Define the linear combinations for A and B blocks
    # A_blocks: 11, 12, 21, 22
    # B_blocks: 11, 12, 21, 22
    
    # P1: A=11+22, B=11+22
    # P2: A=21+22, B=11
    # P3: A=11, B=12-22
    # P4: A=22, B=21-11
    # P5: A=11+12, B=22
    # P6: A=21-11, B=11+12
    # P7: A=12-22, B=21+22
    
    # Output C blocks:
    # C11 = P1 + P4 - P3 - P7
    # C12 = P5 + P7
    # C21 = P6 - P4
    # C22 = P2 + P3 - P6 - P7
    
    # We iterate through the 7 products.
    # For each product, we add rank_sub terms.
    
    for k in range(7):
        # Determine which A blocks and B blocks are involved and their signs
        # A_blocks: list of (block_idx, sign)
        # B_blocks: list of (block_idx, sign)
        # C_blocks: list of (block_idx, sign)
        
        if k == 0: # P1
            A_blocks = [(0, 1), (3, 1)]
            B_blocks = [(0, 1), (3, 1)]
            C_blocks = [(0, 1)] # C11
        elif k == 1: # P2
            A_blocks = [(2, 1), (3, 1)]
            B_blocks = [(0, 1)]
            C_blocks = [(3, 1)] # C22
        elif k == 2: # P3
            A_blocks = [(0, 1)]
            B_blocks = [(1, 1), (3, -1)]
            C_blocks = [(0, -1), (3, 1)] # C11, C22
        elif k == 3: # P4
            A_blocks = [(3, 1)]
            B_blocks = [(2, 1), (0, -1)]
            C_blocks = [(0, 1), (2, -1)] # C11, C21
        elif k == 4: # P5
            A_blocks = [(0, 1), (1, 1)]
            B_blocks = [(3, 1)]
            C_blocks = [(1, 1)] # C12
        elif k == 5: # P6
            A_blocks = [(2, 1), (0, -1)]
            B_blocks = [(0, 1), (1, 1)]
            C_blocks = [(2, 1), (3, -1)] # C21, C22
        else: # P7
            A_blocks = [(1, 1), (3, -1)]
            B_blocks = [(2, 1), (3, 1)]
            C_blocks = [(0, -1), (1, 1), (3, -1)] # C11, C12, C22

        for r in range(rank_sub):
            u_sub = U_sub[r]
            v_sub = V_sub[r]
            w_sub = W_sub[r]
            
            u_full = [0] * (n*n)
            v_full = [0] * (n*n)
            w_full = [0] * (n*n)
            
            # Expand A
            for b_idx, sign in A_blocks:
                # b_idx 0: 11, 1: 12, 2: 21, 3: 22
                # Map sub indices to full indices
                for i in range(m):
                    for j in range(m):
                        sub_idx = i*m + j
                        if b_idx == 0: # 11
                            full_idx = i*m + j
                        elif b_idx == 1: # 12
                            full_idx = i*m + (j+m)
                        elif b_idx == 2: # 21
                            full_idx = (i+m)*m + j
                        else: # 22
                            full_idx = (i+m)*m + (j+m)
                        u_full[full_idx] += sign * u_sub[sub_idx]
                        
            # Expand B
            for b_idx, sign in B_blocks:
                for i in range(m):
                    for j in range(m):
                        sub_idx = i*m + j
                        if b_idx == 0: # 11
                            full_idx = i*m + j
                        elif b_idx == 1: # 12
                            full_idx = i*m + (j+m)
                        elif b_idx == 2: # 21
                            full_idx = (i+m)*m + j
                        else: # 22
                            full_idx = (i+m)*m + (j+m)
                        v_full[full_idx] += sign * v_sub[sub_idx]
                        
            # Expand C
            for b_idx, sign in C_blocks:
                for i in range(m):
                    for j in range(m):
                        sub_idx = i*m + j
                        if b_idx == 0: # 11
                            full_idx = i*m + j
                        elif b_idx == 1: # 12
                            full_idx = i*m + (j+m)
                        elif b_idx == 2: # 21
                            full_idx = (i+m)*m + j
                        else: # 22
                            full_idx = (i+m)*m + (j+m)
                        w_full[full_idx] += sign * w_sub[sub_idx]
                        
            U.append(u_full)
            V.append(v_full)
            W.append(w_full)
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # U1 (16x49), V1 (16x49), W1 (16x49)
    U1, V1, W1 = _combine_strassen(4)
    
    # Part 2: 4x4x2 Schoolbook (Rank 32)
    # A is 4x4, B is 4x2, C is 4x2
    # Indices: A(i,j), B(j,k), C(i,k)
    # U2 (16x32), V2 (8x32), W2 (8x32)
    # Note: The global V and W dimensions are different.
    # Global V is (m*p) x rank = 30 x rank.
    # Global W is (p*n) x rank = 24 x rank.
    
    # We need to align the indices.
    # For Part 1 (4x4x4):
    # A indices: 0..15 (4x4)
    # B indices: 0..15 (4x4) -> Corresponds to first 4 columns of global B (5x6).
    # C indices: 0..15 (4x4) -> Corresponds to first 4 columns of global C (4x6).
    
    # For Part 2 (4x4x2):
    # A indices: 0..15 (4x4)
    # B indices: 0..7 (4x2) -> Corresponds to first 4 rows, last 2 columns of global B?
    # No, we split B (5x6) into B11 (4x4) and B12 (4x2) and B21 (1x6)?
    # My previous split was:
    # A = [A11 (4x4) | A12 (4x1)]
    # B = [B11 (4x6); B21 (1x6)]
    # C = A11 B11 + A12 B21
    
    # So Part 1 is A11 (4x4) * B11_top (4x4).
    # B11 is 4x6. We split B11 into B11a (4x4) and B11b (4x2).
    # So Part 1 is A11 * B11a.
    # Part 2 is A11 * B11b.
    # Part 3 is A12 * B21.
    
    # Let's re-verify the indices for Part 2.
    # A11 is 4x4.
    # B11b is 4x2.
    # C1b is 4x2.
    # Schoolbook for 4x4x2.
    # Rank 32.
    
    U2 = []
    V2 = []
    W2 = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u = [0] * 16
                v = [0] * 8
                w = [0] * 8
                u[i*4+j] = 1
                v[j*2+k] = 1
                w[i*2+k] = 1
                U2.append(u)
                V2.append(v)
                W2.append(w)
                
    # Part 3: 4x1x6 Rank 1
    # A12 is 4x1.
    # B21 is 1x6.
    # C2 is 4x6.
    # Rank 1.
    U3 = []
    V3 = []
    W3 = []
    u = [0] * 16
    v = [0] * 8
    w = [0] * 8
    # A12 corresponds to A column 4 (index 4).
    # Rows 0..3.
    for i in range(4):
        u[i*4+4] = 1
    # B21 corresponds to B row 4 (index 4).
    # Cols 0..5.
    for k in range(6):
        v[4*6+k] = 1
    # C2 corresponds to C rows 0..3, cols 0..5.
    for i in range(4):
        for k in range(6):
            w[i*6+k] = 1
    U3.append(u)
    V3.append(v)
    W3.append(w)
    
    # Combine
    rank = len(U1) + len(U2) + len(U3)
    
    U = [[0] * rank for _ in range(4*5)] # 20 x rank
    V = [[0] * rank for _ in range(5*6)] # 30 x rank
    W = [[0] * rank for _ in range(6*4)] # 24 x rank
    
    col = 0
    # Add Part 1
    for r in range(len(U1)):
        for i in range(16):
            U[i][col] = U1[r][i]
        for j in range(16):
            # B11a is top-left 4x4 of B.
            # Global B index: j (0..15)
            V[j][col] = V1[r][j]
        for k in range(16):
            # C1a is top-left 4x4 of C.
            # Global C index: k (0..15)
            W[k][col] = W1[r][k]
        col += 1
        
    # Add Part 2
    for r in range(len(U2)):
        for i in range(16):
            U[i][col] = U2[r][i]
        for j in range(8):
            # B11b is top-right 4x2 of B.
            # Global B index: 16 + j
            V[16+j][col] = V2[r][j]
        for k in range(8):
            # C1b is top-right 4x2 of C.
            # Global C index: 16 + k
            W[16+k][col] = W2[r][k]
        col += 1
        
    # Add Part 3
    for r in range(len(U3)):
        for i in range(16):
            U[i][col] = U3[r][i]
        for j in range(8):
            # V3 is 1x6, but stored in 8 dims?
            # My V3 construction used 8 dims.
            # B21 is row 4 of B.
            # Global B index: 24 + k?
            # B is 5x6.
            # Rows 0-3 are B11 (24 entries).
            # Row 4 is B21 (6 entries).
            # Indices 24..29.
            # My V3 had v[4*6+k] = 1. 24+k. Correct.
            V[24+j][col] = V3[r][j] if j < 6 else 0
        for k in range(8):
            # W3 is 4x6.
            # Global C index: k (0..23).
            # My W3 had w[i*6+k] = 1.
            W[k][col] = W3[r][k] if k < 24 else 0
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
