import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Rank-minimal decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy:
    Split m=4 into two blocks of 2.
    For each 2x2x7 block, split p=7 into three 2x2x2 blocks and one 2x2x1 block.
    Use Strassen (rank 7) for 2x2x2 blocks.
    Use Schoolbook (rank 4) for 2x2x1 block.
    
    Per 2x2x7 block: 3 * 7 + 4 = 25.
    Total rank: 2 * 25 = 50.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    total_rank = 50
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 8 x 50
    V = np.zeros((m * p, total_rank), dtype=float)  # 28 x 50
    W = np.zeros((p * n, total_rank), dtype=float)  # 14 x 50
    
    col = 0
    
    def add_rank1_term(U_idx, V_idx, W_idx, coeff):
        nonlocal col
        U[U_idx, col] += coeff
        V[V_idx, col] += coeff
        W[W_idx, col] += coeff
        col += 1

    # Split m=4 into two blocks of size 2
    for l in range(2):
        j_base = l * 2
        
        # Split p=7 into three 2x2 blocks and one 2x1 block
        # Blocks 0, 1, 2 are 2x2
        for k_block in range(3):
            k_base = k_block * 2
            
            # Strassen 2x2x2
            # A indices: i in {0,1}, j in {0,1} relative to j_base
            # B indices: j in {0,1} relative to j_base, k in {0,1} relative to k_base
            # C indices: k in {0,1} relative to k_base, i in {0,1}
            
            # M1 = (A11 + A22) * (B11 + B22)
            # U: A11, A22
            # V: B11, B22
            # W: C11, C22 (with signs)
            
            # We need to represent the products.
            # Strassen terms are products of linear combinations.
            # We can expand each linear combination into rank-1 terms?
            # No, Strassen reduces the number of multiplications.
            # Each multiplication is a rank-1 term in the tensor decomposition?
            # Yes, $M_r = ( \sum u_i A_i ) ( \sum v_j B_j )$.
            # This corresponds to a rank-1 tensor term with U and V having multiple non-zero entries.
            
            # Term 1: (A00 + A11) * (B00 + B11)
            # U: U[0, j0] = 1, U[1, j1] = 1 (where j0=0, j1=1 relative)
            # V: V[j0, k0] = 1, V[j1, k1] = 1
            # W: W[k0, 0] = 1, W[k1, 1] = 1 (C00 and C11 get +1)
            # Wait, C00 = M1... C11 = M1...
            # So W[k0, 0] += 1, W[k1, 1] += 1.
            
            # Let's define a helper to add a Strassen term.
            def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
                # u_coeffs: dict { (i, j_rel): coeff }
                # v_coeffs: dict { (j_rel, k_rel): coeff }
                # w_coeffs: dict { (k_rel, i): coeff }
                nonlocal col
                for (i, j_rel), c_u in u_coeffs.items():
                    U[i * m + (j_base + j_rel), col] += c_u
                for (j_rel, k_rel), c_v in v_coeffs.items():
                    V[(j_base + j_rel) * p + (k_base + k_rel), col] += c_v
                for (k_rel, i), c_w in w_coeffs.items():
                    W[(k_base + k_rel) * n + i, col] += c_w
                col += 1

            # M1: (A00 + A11) * (B00 + B11)
            add_strassen_term(
                {(0, 0): 1, (1, 1): 1},
                {(0, 0): 1, (1, 1): 1},
                {(0, 0): 1, (1, 1): 1}
            )
            
            # M2: (A10 + A11) * B00
            add_strassen_term(
                {(1, 0): 1, (1, 1): 1},
                {(0, 0): 1},
                {(1, 0): 1, (0, 0): 1} # C10 += M2, C11 += M2? 
                # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
                # So C10 (i=1, k=0) gets M2. C11 (i=1, k=1) gets -M2.
                # Wait, C indices in Strassen:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # So M2 contributes to C10 (+1) and C11 (-1).
            )
            # Correction: W coeffs for M2
            add_strassen_term(
                {(1, 0): 1, (1, 1): 1},
                {(0, 0): 1},
                {(0, 1): 1, (1, 1): -1} # k_rel=0 -> C10, k_rel=1 -> C11
            )
            
            # M3: A00 * (B10 - B11)
            add_strassen_term(
                {(0, 0): 1},
                {(1, 0): 1, (1, 1): -1},
                {(0, 0): 1, (1, 0): 1} # C01 += M3, C11 += M3
            )
            
            # M4: A11 * (B01 - B00)
            add_strassen_term(
                {(1, 1): 1},
                {(0, 1): 1, (0, 0): -1},
                {(0, 0): 1, (0, 1): 1} # C00 += M4, C10 += M4
            )
            
            # M5: (A00 + A01) * B11
            add_strassen_term(
                {(0, 0): 1, (0, 1): 1},
                {(1, 1): 1},
                {(0, 0): -1, (0, 1): 1} # C00 -= M5, C01 += M5
            )
            
            # M6: (A10 - A00) * (B00 + B01)
            add_strassen_term(
                {(1, 0): 1, (0, 0): -1},
                {(0, 0): 1, (0, 1): 1},
                {(1, 1): 1} # C11 += M6
            )
            
            # M7: (A01 - A11) * (B10 + B11)
            add_strassen_term(
                {(0, 1): 1, (1, 1): -1},
                {(1, 0): 1, (1, 1): 1},
                {(0, 0): 1, (1, 1): 1} # C00 += M7, C11 += M7
            )
            
        # Block 3 is 2x1
        k_base = 6
        # Schoolbook 2x2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 B0
        add_strassen_term(
            {(0, 0): 1},
            {(0, 0): 1},
            {(0, 0): 1}
        )
        # Term 2: A01 B1
        add_strassen_term(
            {(0, 1): 1},
            {(1, 0): 1},
            {(0, 0): 1}
        )
        # Term 3: A10 B0
        add_strassen_term(
            {(1, 0): 1},
            {(0, 0): 1},
            {(0, 1): 1}
        )
        # Term 4: A11 B1
        add_strassen_term(
            {(1, 1): 1},
            {(1, 0): 1},
            {(0, 1): 1}
        )
        
    if col != total_rank:
        print(f"Warning: col {col} != total_rank {total_rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

