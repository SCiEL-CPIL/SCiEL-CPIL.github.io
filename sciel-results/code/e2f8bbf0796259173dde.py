import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank is 98 (two 4x4x4 multiplications of rank 49)
    R = 98
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Strassen 2x2 decomposition (rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j, l], V2[j*2+k, l], W2[k*2+i, l]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # M1
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1
    
    # M2
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1; S_W[3, 1] = 1
    
    # M3
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1; S_W[3, 2] = 1
    
    # M4
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1
    
    # M5
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1; S_W[1, 4] = 1
    
    # M6
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1
    
    # M7
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = 1
    
    # Define the 7 super-terms for 4x4
    # Each term is (A_coeffs, B_coeffs)
    # A_coeffs is a 2x2 matrix of coefficients for A00, A01, A10, A11
    # B_coeffs is a 2x2 matrix of coefficients for B00, B01, B10, B11
    
    # Note: A00 is top-left, A01 top-right, A10 bottom-left, A11 bottom-right
    # Global indices for A (4x4):
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Global indices for B (4x4) for the first half (k=0..3):
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # Global indices for C (4x4) for the first half (k=0..3):
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # Super-term definitions
    # M0: (A00+A11)(B00+B11)
    # M1: (A10+A11)B00
    # M2: A00(B01-B11)
    # M3: A11(B10-B00)
    # M4: (A00+A01)B11
    # M5: (A10-A00)(B00+B01)
    # M6: (A01-A11)(B10+B11)
    
    super_terms = [
        # (A_coeffs, B_coeffs)
        # A_coeffs: [[A00, A01], [A10, A11]]
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])), # M0
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]])), # M1
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]])), # M2
        (np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, -1]])), # M3
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])), # M4
        (np.array([[1, 0], [-1, 0]]), np.array([[1, 1], [0, 0]])), # M5
        (np.array([[0, 1], [0, -1]]), np.array([[0, 0], [1, 1]])), # M6
    ]
    
    # Output combination for 4x4 Strassen
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # We need to map the local 2x2 result of each super-term to the global 4x4 result.
    # For super-term s, the local result is a 2x2 matrix M_s.
    # This M_s is added to the global C with certain coefficients.
    # Actually, the local result M_s IS the block that gets added.
    # But the local result is 2x2. The global C blocks are 2x2.
    # So M_s is added to C00, C01, etc. with coefficients.
    
    # Wait, the Strassen 4x4 algorithm computes M0..M6.
    # Then C is formed by combining M0..M6.
    # So for each s, the "W" part of the tensor decomposition is the linear map that takes the local 2x2 result and adds it to the global C.
    # The local 2x2 result has indices (k_l, i_l) in 0..1.
    # The global C has indices (k, i) in 0..3.
    # The mapping depends on s.
    
    # For s=0 (M0), it contributes to C00 and C11.
    # C00 += M0, C11 += M0.
    # So for a local index (k_l, i_l), the global indices are:
    # For C00: k = k_l, i = i_l.
    # For C11: k = 2 + k_l, i = 2 + i_l.
    # So W_global[k, i] += W_local[k_l, i_l] for both (k,i) in C00 and C11.
    
    # Let's define the W mapping for each s.
    # W_map[s] is a list of (coeff, block_row, block_col)
    # block_row, block_col in 0..1.
    
    W_maps = [
        [(1, 0, 0), (1, 1, 1)], # M0 -> C00, C11
        [(1, 1, 0), (1, 1, 1)], # M1 -> C10, C11
        [(1, 0, 1), (1, 1, 1)], # M2 -> C01, C11
        [(1, 0, 0), (1, 1, 0)], # M3 -> C00, C10
        [(-1, 0, 0), (1, 0, 1)], # M4 -> -C00, C01
        [(1, 1, 1)], # M5 -> C11
        [(-1, 0, 0), (-1, 1, 1)], # M6 -> -C00, -C11
    ]
    
    # Now, build the 49 terms for the first half (B cols 0..3)
    for s in range(7):
        A_coeffs, B_coeffs = super_terms[s]
        for l in range(7):
            col = s * 7 + l
            
            # U part
            # U_local is S_U[:, l] (4x1)
            # Global U index: i*4 + j
            # A_coeffs[i_block, j_block] is coeff for A_{i_block, j_block}
            # A_{i_block, j_block} has local indices i_l, j_l in 0..1
            # Global i = i_block * 2 + i_l
            # Global j = j_block * 2 + j_l
            
            for i_b in range(2):
                for j_b in range(2):
                    coeff_A = A_coeffs[i_b, j_b]
                    if coeff_A != 0:
                        for i_l in range(2):
                            for j_l in range(2):
                                local_u_idx = i_l * 2 + j_l
                                coeff_u = S_U[local_u_idx, l]
                                if coeff_u != 0:
                                    global_i = i_b * 2 + i_l
                                    global_j = j_b * 2 + j_l
                                    global_u_idx = global_i * 4 + global_j
                                    U[global_u_idx, col] += coeff_A * coeff_u
            
            # V part
            # Similar for B
            for i_b in range(2):
                for j_b in range(2):
                    coeff_B = B_coeffs[i_b, j_b]
                    if coeff_B != 0:
                        for i_l in range(2):
                            for j_l in range(2):
                                local_v_idx = i_l * 2 + j_l
                                coeff_v = S_V[local_v_idx, l]
                                if coeff_v != 0:
                                    global_i = i_b * 2 + i_l
                                    global_j = j_b * 2 + j_l
                                    global_v_idx = global_i * 4 + global_j
                                    V[global_v_idx, col] += coeff_B * coeff_v
            
            # W part
            # W_local is S_W[:, l] (4x1)
            # Global W index: k*4 + i
            # For each (coeff, block_row, block_col) in W_maps[s]:
            # local k_l, i_l
            # global k = block_row * 2 + k_l
            # global i = block_col * 2 + i_l
            for coeff_w, block_row, block_col in W_maps[s]:
                for k_l in range(2):
                    for i_l in range(2):
                        local_w_idx = k_l * 2 + i_l
                        coeff_local_w = S_W[local_w_idx, l]
                        if coeff_local_w != 0:
                            global_k = block_row * 2 + k_l
                            global_i = block_col * 2 + i_l
                            global_w_idx = global_k * 4 + global_i
                            W[global_w_idx, col] += coeff_w * coeff_local_w
    
    # Now, duplicate for the second half (B cols 4..7)
    # The U part is the same.
    # The V part is shifted by 4 in the column index of B.
    # The W part is shifted by 4 in the column index of C.
    
    for col in range(49):
        U[:, col + 49] = U[:, col]
        V[:, col + 49] = V[:, col]
        W[:, col + 49] = W[:, col]
        
        # Shift V indices: B is 4x8.
        # V index is j*8 + k.
        # In the first half, k was 0..3.
        # In the second half, k should be 4..7.
        # So we need to add 4 to the k part of the V index.
        # V index = j*8 + k.
        # If we just copy V, the indices are j*4 + k (for 4x4).
        # Wait, my V construction above used global_v_idx = global_i * 4 + global_j.
        # This assumes B is 4x4.
        # But B is 4x8.
        # So I should have used global_v_idx = global_i * 8 + global_j.
        # And for the second half, global_j should be 4..7.
        
        # Let's re-evaluate.
        # I will reconstruct V and W correctly for 4x8.
        
    # The previous construction of V and W was for 4x4 B and C.
    # I need to fix the indices.
    # V shape is (m*p, R) = (4*8, R) = (32, R).
    # W shape is (p*n, R) = (8*4, R) = (32, R).
    
    # Let's restart the construction with correct indices.
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for s in range(7):
        A_coeffs, B_coeffs = super_terms[s]
        for l in range(7):
            col = s * 7 + l
            
            # U part (A is 4x4, index i*4+j)
            for i_b in range(2):
                for j_b in range(2):
                    coeff_A = A_coeffs[i_b, j_b]
                    if coeff_A != 0:
                        for i_l in range(2):
                            for j_l in range(2):
                                local_u_idx = i_l * 2 + j_l
                                coeff_u = S_U[local_u_idx, l]
                                if coeff_u != 0:
                                    global_i = i_b * 2 + i_l
                                    global_j = j_b * 2 + j_l
                                    global_u_idx = global_i * 4 + global_j
                                    U[global_u_idx, col] += coeff_A * coeff_u
            
            # V part (B is 4x8, index j*8+k)
            # First half: k = 0..3
            for i_b in range(2):
                for j_b in range(2):
                    coeff_B = B_coeffs[i_b, j_b]
                    if coeff_B != 0:
                        for i_l in range(2):
                            for j_l in range(2):
                                local_v_idx = i_l * 2 + j_l
                                coeff_v = S_V[local_v_idx, l]
                                if coeff_v != 0:
                                    global_i = i_b * 2 + i_l
                                    global_j = j_b * 2 + j_l
                                    # k = global_j (0..3)
                                    global_v_idx = global_i * 8 + global_j
                                    V[global_v_idx, col] += coeff_B * coeff_v
                                    
                                    # Second half: k = 4..7
                                    # global_j_second = global_j + 4
                                    global_v_idx_second = global_i * 8 + (global_j + 4)
                                    V[global_v_idx_second, col + 49] += coeff_B * coeff_v
            
            # W part (C is 8x4, index k*4+i)
            # First half: k = 0..3
            for coeff_w, block_row, block_col in W_maps[s]:
                for k_l in range(2):
                    for i_l in range(2):
                        local_w_idx = k_l * 2 + i_l
                        coeff_local_w = S_W[local_w_idx, l]
                        if coeff_local_w != 0:
                            global_k = block_row * 2 + k_l
                            global_i = block_col * 2 + i_l
                            # k = global_k (0..3)
                            global_w_idx = global_k * 4 + global_i
                            W[global_w_idx, col] += coeff_w * coeff_local_w
                            
                            # Second half: k = 4..7
                            global_k_second = global_k + 4
                            global_w_idx_second = global_k_second * 4 + global_i
                            W[global_w_idx_second, col + 49] += coeff_w * coeff_local_w
            
            # U part for second half is the same
            U[:, col + 49] = U[:, col]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
