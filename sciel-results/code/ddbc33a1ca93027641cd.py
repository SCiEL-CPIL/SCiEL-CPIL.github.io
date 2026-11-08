import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x3 Schoolbook (Rank 48)
    
    # Helper: Generate 2x2x2 Strassen matrices
    def get_strassen_222():
        U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ], dtype=float).T # Shape 4x7
        V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=float).T # Shape 4x7
        W = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [1, -1, 1, 0]
        ], dtype=float).T # Shape 4x7
        return U, V, W

    U_222, V_222, W_222 = get_strassen_222()
    
    # Define block mappings for 4x4x4
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # Indices in U (A side): i*4 + j
    # Block (br, bc) covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will construct U_444, V_444, W_444
    # Rank 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Block index to global index mapping
    def block_to_indices(br, bc):
        rows = [2*br, 2*br+1]
        cols = [2*bc, 2*bc+1]
        return [(r, c) for r in rows for c in cols]
        
    # Strassen M definitions
    # Each M is a product of P (from A) and Q (from B)
    # P and Q are 2x2 matrices of blocks
    
    # M1
    P1_blocks = [(0,0,1), (1,1,1)] # (br, bc, coeff)
    Q1_blocks = [(0,0,1), (1,1,1)]
    # M2
    P2_blocks = [(1,0,1), (1,1,1)]
    Q2_blocks = [(0,0,1)]
    # M3
    P3_blocks = [(0,0,1)]
    Q3_blocks = [(0,1,1), (1,1,-1)]
    # M4
    P4_blocks = [(1,1,1)]
    Q4_blocks = [(1,0,1), (0,0,-1)]
    # M5
    P5_blocks = [(0,0,1), (0,1,1)]
    Q5_blocks = [(1,1,1)]
    # M6
    P6_blocks = [(1,0,1), (0,0,-1)]
    Q6_blocks = [(0,0,1), (0,1,1)]
    # M7
    P7_blocks = [(0,1,1), (1,1,-1)]
    Q7_blocks = [(1,0,1), (1,1,1)]
    
    M_defs = [
        (P1_blocks, Q1_blocks),
        (P2_blocks, Q2_blocks),
        (P3_blocks, Q3_blocks),
        (P4_blocks, Q4_blocks),
        (P5_blocks, Q5_blocks),
        (P6_blocks, Q6_blocks),
        (P7_blocks, Q7_blocks)
    ]
    
    for m_idx, (P_blocks, Q_blocks) in enumerate(M_defs):
        # P is 2x2 matrix of blocks
        # Q is 2x2 matrix of blocks
        # We need to form the 2x2 matrix P and Q in terms of global A/B indices
        
        # Initialize local 2x2 matrices for P and Q
        # P_local[0,0] = sum of coeffs for block (0,0) in P_blocks
        # But P_blocks are sparse.
        
        # Actually, it's easier to just iterate over the 7 sub-products of the 2x2 Strassen
        # For each sub-product s in 0..6:
        # U_222[:, s] gives coefficients for A_local indices (0,1,2,3)
        # V_222[:, s] gives coefficients for B_local indices (0,1,2,3)
        # W_222[:, s] gives coefficients for C_local indices (0,1,2,3)
        
        for s in range(7):
            col_offset = m_idx * 7 + s
            
            # Get coefficients for A_local (0..3)
            a_coeffs = U_222[:, s]
            # Get coefficients for B_local (0..3)
            b_coeffs = V_222[:, s]
            # Get coefficients for C_local (0..3)
            c_coeffs = W_222[:, s]
            
            # Map A_local indices to global A indices using P_blocks
            # A_local index 0 -> P[0,0]
            # A_local index 1 -> P[0,1]
            # A_local index 2 -> P[1,0]
            # A_local index 3 -> P[1,1]
            
            # P[0,0] is sum of blocks in P_blocks with br=0, bc=0
            # P[0,1] is sum of blocks in P_blocks with br=0, bc=1
            # P[1,0] is sum of blocks in P_blocks with br=1, bc=0
            # P[1,1] is sum of blocks in P_blocks with br=1, bc=1
            
            # We need to distribute the coefficient a_coeffs[l] to the global indices
            # corresponding to P[l//2, l%2]
            
            # Build mapping from local P index to list of (global_idx, coeff)
            # P_local_idx 0 -> (0,0)
            # P_local_idx 1 -> (0,1)
            # P_local_idx 2 -> (1,0)
            # P_local_idx 3 -> (1,1)
            
            for l in range(4):
                coeff_a = a_coeffs[l]
                if coeff_a == 0: continue
                
                br = l // 2
                bc = l % 2
                
                # Find all blocks in P_blocks matching (br, bc)
                for (b_br, b_bc, b_coeff) in P_blocks:
                    if b_br == br and b_bc == bc:
                        # Add to global U
                        global_indices = block_to_indices(br, bc)
                        for gi in global_indices:
                            global_row = gi[0] * 4 + gi[1]
                            U_444[global_row, col_offset] += coeff_a * b_coeff
            
            # Similarly for B
            for l in range(4):
                coeff_b = b_coeffs[l]
                if coeff_b == 0: continue
                
                br = l // 2
                bc = l % 2
                
                for (b_br, b_bc, b_coeff) in Q_blocks:
                    if b_br == br and b_bc == bc:
                        global_indices = block_to_indices(br, bc)
                        for gi in global_indices:
                            global_row = gi[0] * 4 + gi[1]
                            V_444[global_row, col_offset] += coeff_b * b_coeff
            
            # For W (C side)
            # C_local indices map to C blocks
            # C_local 0 -> C[0,0]
            # C_local 1 -> C[0,1]
            # C_local 2 -> C[1,0]
            # C_local 3 -> C[1,1]
            
            # W_222[:, s] are coefficients for C_local
            # We need to map C_local to global C indices
            # C[br, bc] corresponds to block (br, bc) of C
            # C is 4x4, so block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
            
            for l in range(4):
                coeff_c = c_coeffs[l]
                if coeff_c == 0: continue
                
                br = l // 2
                bc = l % 2
                
                global_indices = block_to_indices(br, bc)
                for gi in global_indices:
                    global_row = gi[0] * 4 + gi[1]
                    W_444[global_row, col_offset] += coeff_c
    
    # Part 2: 4x4x3 Schoolbook
    # n=4, m=4, p=3
    # Rank 48
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((12, 48)) # m*p = 4*3
    W_443 = np.zeros((12, 48)) # p*n = 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_443[i * 4 + j, col] = 1
                V_443[j * 3 + k, col] = 1
                W_443[k * 4 + i, col] = 1
                col += 1
                
    # Combine
    # U: 16 x 97
    U = np.hstack([U_444, U_443])
    
    # V: 28 x 97
    # First 16 rows from V_444 (k=0..3)
    # Next 12 rows from V_443 (k=4..6)
    V = np.zeros((28, 97))
    # Map V_444 rows (j*4+k) to global (j*7+k)
    for j in range(4):
        for k in range(4):
            local_row = j * 4 + k
            global_row = j * 7 + k
            V[global_row, :49] = V_444[local_row, :]
            
    # Map V_443 rows (j*3+k_local) to global (j*7 + (4+k_local))
    for j in range(4):
        for k_local in range(3):
            local_row = j * 3 + k_local
            global_row = j * 7 + (4 + k_local)
            V[global_row, 49:] = V_443[local_row, :]
            
    # W: 28 x 97
    # First 16 rows from W_444 (k=0..3)
    # Next 12 rows from W_443 (k=4..6)
    W = np.zeros((28, 97))
    # Map W_444 rows (k*4+i) to global (k*4+i)
    for k in range(4):
        for i in range(4):
            local_row = k * 4 + i
            global_row = k * 4 + i
            W[global_row, :49] = W_444[local_row, :]
            
    # Map W_443 rows (k_local*4+i) to global ((4+k_local)*4+i)
    for k_local in range(3):
        for i in range(4):
            local_row = k_local * 4 + i
            global_row = (4 + k_local) * 4 + i
            W[global_row, 49:] = W_443[local_row, :]
            
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
