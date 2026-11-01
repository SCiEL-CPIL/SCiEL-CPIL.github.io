"""Matrix multiplication 2,2,14 - Strassen-based rank-49 decomposition."""
import time

DIMENSIONS = [2, 2, 14]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with safety margin
    
    n, m, p = DIMENSIONS
    # p = 14 = 7 blocks of 2 columns each
    num_blocks = p // 2  # 7
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Tensor convention: T[i*m+j, j*p+k, k*n+i] = 1
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (28, rank)  
    # W has shape (p*n, rank) = (28, rank)
    # W index: k*n + i
    
    # Strassen factors for a single 2x2*2x2 block:
    # For A[i,j] and B[j,l] where l is 0 or 1 within the block
    # U factors (for A): each m_r is a linear combo of A entries
    # V factors (for B): each m_r is a linear combo of B entries  
    # W factors (for C): each C entry is a linear combo of m_r values
    
    # For one block at column offset k0 (k0 = 2*b):
    # B entries: B[j, k0+l] for j in {0,1}, l in {0,1}
    # V index for B[j, k0+l] is j*p + (k0+l)
    # W index for C[i, k0+l] is (k0+l)*n + i
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 1, 0],  # placeholder - we'll build per product
    ]
    
    # Let me define Strassen properly for 2x2*2x2:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # A indices: i*m+j -> A00:0, A01:1, A10:2, A11:3
    # B indices: j*p+l -> B00:0, B01:1, B10:2, B11:3 (within block)
    # C indices: l*n+i -> C00:0, C01:1, C10:2, C11:3 (within block)
    
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11): U=[1,0,0,1], V=[1,0,0,1]
    # M2 = (A10+A11)*B00: U=[0,0,1,1], V=[1,0,0,0]
    # M3 = A00*(B01-B11): U=[1,0,0,0], V=[0,1,0,-1]
    # M4 = A11*(B10-B00): U=[0,0,0,1], V=[0,0,1,-1]
    # M5 = (A00+A10)*B11: U=[1,0,1,0], V=[0,0,0,1]
    # M6 = (A01-A00)*(B00+B10): U=[-1,1,0,0], V=[1,0,1,0]
    # M7 = (A10-A11)*(B01+B11): U=[0,0,1,-1], V=[0,1,0,1]
    
    # C reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_defs = [
        # (U_coeffs for A00,A01,A10,A11, V_coeffs for B00,B01,B10,B11)
        ([1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1]),  # M4
        ([1, 0, 1, 0], [0, 0, 0, 1]),   # M5
        ([-1, 1, 0, 0], [1, 0, 1, 0]),  # M6
        ([0, 0, 1, -1], [0, 1, 0, 1]),  # M7
    ]
    
    # C[i_block, l] = sum_r c_{i_block,l,r} * M_r
    # i_block in {0,1} (row of C), l in {0,1} (col within block)
    # C index within block: l*n + i_block
    c_coeffs = [
        # For C00 (i=0, l=0): M1 + M4 - M5 + M7
        [1, 0, 0, 1, -1, 0, 1],
        # For C01 (i=0, l=1): M3 + M5
        [0, 0, 1, 0, 1, 0, 0],
        # For C10 (i=1, l=0): M2 + M4
        [0, 1, 0, 1, 0, 0, 0],
        # For C11 (i=1, l=1): M1 - M2 + M3 + M6
        [1, -1, 1, 0, 0, 1, 0],
    ]
    # c_coeffs[i_block, l][r] where C index = l*n + i_block
    # So c_coeffs indexed by [i_block][l]
    # Let me re-index: c_coeff[i_block][l][r]
    # C index in block: l*2 + i_block
    c_coeff = {}
    for i_block in range(2):
        for l in range(2):
            c_idx = l * 2 + i_block
            c_coeff[(i_block, l)] = c_coeffs[c_idx]
    
    rank = num_blocks * 7  # 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    for b in range(num_blocks):
        k0 = 2 * b  # starting column of this block
        for r in range(7):
            col = b * 7 + r
            u_coeffs, v_coeffs = strassen_defs[r]
            
            # Fill U: A indices i*m+j
            for i in range(2):
                for j in range(2):
                    a_idx = i * m + j
                    U[a_idx][col] = u_coeffs[i * 2 + j]
            
            # Fill V: B indices j*p + k
            for j in range(2):
                for l in range(2):
                    k = k0 + l
                    v_idx = j * p + k
                    V[v_idx][col] = v_coeffs[j * 2 + l]
            
            # Fill W: C indices k*n + i
            for i_block in range(2):
                for l in range(2):
                    k = k0 + l
                    w_idx = k * n + i_block
                    W[w_idx][col] = c_coeff[(i_block, l)][r]
    
    return {"U": U, "V": V, "W": W}
