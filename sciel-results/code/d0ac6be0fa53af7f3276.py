import numpy as np
import time

# Dimensions
N = 4
M = 4
P = 4
RANK = 49

def construct():
    """
    Construct the rank-49 tensor decomposition for 4x4 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer entries.
    """
    start_time = time.time()
    DEADLINE = 10.0  # Generous deadline, computation is < 0.1s
    
    # Initialize U, V, W as zero matrices
    # U shape: (N*M, RANK) = (16, 49)
    # V shape: (M*P, RANK) = (16, 49)
    # W shape: (P*N, RANK) = (16, 49)
    
    U = np.zeros((N * M, RANK), dtype=float)
    V = np.zeros((M * P, RANK), dtype=float)
    W = np.zeros((P * N, RANK), dtype=float)
    
    # Strassen 2x2 decomposition terms
    # Each term is defined by linear forms for X (input A part), Y (input B part), and Z (output C part)
    # X, Y, Z are 2x2 matrices.
    # We represent the linear form as a list of (row, col, coeff) tuples for the 2x2 block.
    
    # The 7 Strassen products for 2x2 matrices:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # Output C = A * B:
    # c00 = m1 - m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Define the 7 terms for a single 2x2 multiplication.
    strassen_2x2 = [
        # m1
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # m2
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1), (1, 1, -1)]
        ),
        # m3
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)]
        ),
        # m4
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        ),
        # m5
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (0, 1, 1)]
        ),
        # m6
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # m7
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]
    
    # Block-level Strassen products for 4x4 matrices
    block_strassen = [
        # M1 = (A00 + A11) * (B00 + B11)
        (
            [(0, 0, 1), (1, 1, 1)],  # X blocks
            [(0, 0, 1), (1, 1, 1)],  # Y blocks
            [(0, 0, 1), (1, 1, 1)]   # Z blocks
        ),
        # M2 = (A10 + A11) * B00
        (
            [(1, 0, 1), (1, 1, 1)],  # X blocks
            [(0, 0, 1)],             # Y blocks
            [(1, 0, 1), (1, 1, -1)]  # Z blocks
        ),
        # M3 = A00 * (B01 - B11)
        (
            [(0, 0, 1)],             # X blocks
            [(0, 1, 1), (1, 1, -1)], # Y blocks
            [(0, 1, 1), (1, 1, 1)]   # Z blocks
        ),
        # M4 = A11 * (B10 - B00)
        (
            [(1, 1, 1)],             # X blocks
            [(1, 0, 1), (0, 0, -1)], # Y blocks
            [(0, 0, 1), (1, 0, 1)]   # Z blocks
        ),
        # M5 = (A00 + A01) * B11
        (
            [(0, 0, 1), (0, 1, 1)],  # X blocks
            [(1, 1, 1)],             # Y blocks
            [(0, 0, -1), (0, 1, 1)]  # Z blocks
        ),
        # M6 = (A10 - A00) * (B00 + B01)
        (
            [(1, 0, 1), (0, 0, -1)], # X blocks
            [(0, 0, 1), (0, 1, 1)],  # Y blocks
            [(1, 1, 1)]              # Z blocks
        ),
        # M7 = (A01 - A11) * (B10 + B11)
        (
            [(0, 1, 1), (1, 1, -1)], # X blocks
            [(1, 0, 1), (1, 1, 1)],  # Y blocks
            [(0, 0, 1)]              # Z blocks
        ),
    ]
    
    # Helper to get flat index for a 2x2 block element in a 4x4 matrix
    def block_elem_to_flat(block_r, block_c, elem_r, elem_c):
        row = 2 * block_r + elem_r
        col = 2 * block_c + elem_c
        return row * 4 + col
    
    term_idx = 0
    for b_idx, (X_blocks, Y_blocks, Z_blocks) in enumerate(block_strassen):
        for s_idx, (x_form, y_form, z_form) in enumerate(strassen_2x2):
            if time.time() - start_time > DEADLINE:
                # Fallback to schoolbook if somehow we hit deadline (unlikely)
                U_school = np.zeros((16, 64))
                V_school = np.zeros((16, 64))
                W_school = np.zeros((16, 64))
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            idx = i * 16 + j * 4 + k
                            U_school[i * 4 + j, idx] = 1.0
                            V_school[j * 4 + k, idx] = 1.0
                            W_school[k * 4 + i, idx] = 1.0
                return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

            # U linear form
            for er, ec, coeff_s in x_form:
                for br, bc, coeff_x in X_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    U[flat_idx, term_idx] += coeff_s * coeff_x
            
            # V linear form
            for er, ec, coeff_s in y_form:
                for br, bc, coeff_y in Y_blocks:
                    flat_idx = block_elem_to_flat(br, bc, er, ec)
                    V[flat_idx, term_idx] += coeff_s * coeff_y
            
            # W linear form
            for er, ec, coeff_z in z_form:
                for br, bc, coeff_zb in Z_blocks:
                    i = 2 * br + er
                    k = 2 * bc + ec
                    flat_idx = k * 4 + i
                    W[flat_idx, term_idx] += coeff_z * coeff_zb
            
            term_idx += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
