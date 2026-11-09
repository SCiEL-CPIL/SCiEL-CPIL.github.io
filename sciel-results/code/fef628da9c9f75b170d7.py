import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition in parts:
    # Part 1: 4x4x4 (top-left blocks) using Strassen: rank 49
    # Part 2: 4x4x1 (A_left * B_top_right): rank 16
    # Part 3: 4x1x4 (A_right * B_bottom_left): rank 16
    # Part 4: 4x1x1 (A_right * B_bottom_right): rank 4
    # Total rank: 49 + 16 + 16 + 4 = 85
    
    rank = 85
    U = np.zeros((n * m, rank), dtype=np.int64)  # 20 x 85
    V = np.zeros((m * p, rank), dtype=np.int64)  # 25 x 85
    W = np.zeros((p * n, rank), dtype=np.int64)  # 20 x 85
    
    # Strassen 2x2x2 factors
    # For multiplying 2x2 matrices: C = A * B
    # U indices: (0,0), (0,1), (1,0), (1,1) -> rows 0,1,2,3
    # V indices: (0,0), (0,1), (1,0), (1,1) -> rows 0,1,2,3
    # W indices: (0,0), (0,1), (1,0), (1,1) -> using k*n+i convention with n=2
    #   W row index = k*2 + i, so (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    def get_strassen_2x2():
        """Return U, V, W factors for 2x2x2 Strassen, each 4x7."""
        u = np.zeros((4, 7), dtype=np.int64)
        v = np.zeros((4, 7), dtype=np.int64)
        w = np.zeros((4, 7), dtype=np.int64)
        
        # Term 0: (a00+a11)(b00+b11)
        # Contributes to c00 +=, c11 +=
        u[0,0]=1; u[3,0]=1
        v[0,0]=1; v[3,0]=1
        w[0,0]=1; w[3,0]=1
        
        # Term 1: (a10+a11)b00
        # Contributes to c10 +=, c11 -=
        u[2,1]=1; u[3,1]=1
        v[0,1]=1
        w[2,1]=1; w[3,1]=-1
        
        # Term 2: a00(b01-b11)
        # Contributes to c01 +=, c11 +=
        u[0,2]=1
        v[1,2]=1; v[3,2]=-1
        w[1,2]=1; w[3,2]=1
        
        # Term 3: a11(b10-b00)
        # Contributes to c00 +=, c10 +=
        u[3,3]=1
        v[2,3]=1; v[0,3]=-1
        w[0,3]=1; w[2,3]=1
        
        # Term 4: (a00+a01)b11
        # Contributes to c00 -=, c01 +=
        u[0,4]=1; u[1,4]=1
        v[3,4]=1
        w[0,4]=-1; w[1,4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        # Contributes to c11 +=
        u[0,5]=-1; u[2,5]=1
        v[0,5]=1; v[1,5]=1
        w[3,5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        # Contributes to c00 +=
        u[1,6]=1; u[3,6]=-1
        v[2,6]=1; v[3,6]=1
        w[0,6]=1
        
        return u, v, w
    
    s_u, s_v, s_w = get_strassen_2x2()
    
    # Part 1: 4x4x4 part using Strassen at block level
    # A is partitioned into 2x2 blocks of 2x2 matrices:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bi is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    #
    # Strassen at block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product, decomposed via 2x2 Strassen (7 scalar mults)
    # Total: 7 * 7 = 49 scalar multiplications
    
    # For each block-level Strassen term, we need to:
    # 1. Identify which A blocks and B blocks are involved with what coefficients
    # 2. For each of the 7 scalar Strassen terms within, map to global indices
    
    # A block (br, bc) occupies rows 2*br..2*br+1, cols 2*bc..2*bc+1 in the 4x4
    # Global U row for A[i,j] = i*5 + j (since A is 4x5, but we only use cols 0-3 for this part)
    # Global V row for B[i,j] = i*5 + j (B is 5x5, we use rows 0-3, cols 0-3 for this part)
    # Global W row for C[i,j] = j*4 + i (C is 4x5, using k*n+i convention with n=4)
    
    # Define the 7 block-level Strassen terms
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (br,bc) to coeff for A blocks
    # b_coeffs: dict from (br,bc) to coeff for B blocks
    # c_coeffs: dict from (br,bc) to coeff for C blocks (which C blocks this term contributes to)
    
    block_terms = [
        # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00, contributes to C10 and C11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11), contributes to C01 and C11
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00), contributes to C00 and C10
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11, contributes to C00 and C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01), contributes to C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11), contributes to C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for a_coeffs, b_coeffs, c_coeffs in block_terms:
        for inner_k in range(7):
            col_idx = col
            # Get the 2x2 Strassen factors for this inner term
            u_vec = s_u[:, inner_k]  # coeffs for x00,x01,x10,x11
            v_vec = s_v[:, inner_k]  # coeffs for y00,y01,y10,y11
            w_vec = s_w[:, inner_k]  # coeffs for c00,c01,c10,c11
            
            # U part: for each A block with nonzero coeff
            for (a_br, a_bc), a_coeff in a_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row, col_idx] += a_coeff * u_val
            
            # V part: for each B block with nonzero coeff
            for (b_br, b_bc), b_coeff in b_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr  # row in B
                            global_k = 2 * b_bc + lc  # col in B
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row, col_idx] += b_coeff * v_val
            
            # W part: for each C block with nonzero coeff
            for (c_br, c_bc), c_coeff in c_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_i = 2 * c_br + lr  # row in C
                            global_k = 2 * c_bc + lc  # col in C
                            global_w_row = global_k * 4 + global_i  # k*n+i with n=4
                            W[global_w_row, col_idx] += c_coeff * w_val
            
            col += 1
    
    # col should be 49 now
    assert col == 49, f"Expected col=49, got {col}"
    
    # Part 2: 4x4x1 part: A[:,0:4] * B[0:4,4] -> C[:,4]
    # A is 4x4 (rows 0-3, cols 0-3), B is 4x1 (rows 0-3, col 4), C is 4x1 (rows 0-3, col 4)
    # Trivial decomposition: rank 4*4*1 = 16
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1  # k=4, i=i, so k*4+i = 16+i
            col += 1
    
    # col should be 49+16=65
    assert col == 65, f"Expected col=65, got {col}"
    
    # Part 3: 4x1x4 part: A[:,4] * B[4,0:4] -> C[:,0:4]
    # A is 4x1 (rows 0-3, col 4), B is 1x4 (row 4, cols 0-3), C is 4x4 (rows 0-3, cols 0-3)
    # Trivial decomposition: rank 4*1*4 = 16
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
    
    # col should be 65+16=81
    assert col == 81, f"Expected col=81, got {col}"
    
    # Part 4: 4x1x1 part: A[:,4] * B[4,4] -> C[:,4]
    # A is 4x1 (rows 0-3, col 4), B is 1x1 (row 4, col 4), C is 4x1 (rows 0-3, col 4)
    # Trivial decomposition: rank 4*1*1 = 4
    for i in range(4):
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    # col should be 81+4=85
    assert col == 85, f"Expected col=85, got {col}"
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > DEADLINE:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
