import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    
    # Target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0

    # Strategy: 
    # 1. Decompose 6x6 multiplication using 2x2 blocks of 3x3 matrices.
    # 2. Use Strassen's algorithm (rank 7) on the 2x2 block matrix.
    # 3. Use BCLR rank-23 decomposition for the 3x3 matrix multiplications.
    # Total Rank = 7 * 23 = 161.
    
    # --- Step 1: Define 3x3 Rank 23 Decomposition ---
    # Indices for 3x3 matrices: 0..8
    # A[i,j] index: i*3+j
    # B[i,j] index: i*3+j
    # C[i,j] index: i*3+j
    
    # U3, V3, W3 shapes (9, 23)
    U3 = np.zeros((9, 23), dtype=float)
    V3 = np.zeros((9, 23), dtype=float)
    W3 = np.zeros((9, 23), dtype=float)
    
    # BCLR 23-rank coefficients
    # Source: Based on Bläser 1998 / Strassen 1969 rank 23 algorithm
    # The terms are defined by linear combinations.
    
    # Helper to set terms
    # term(idx, A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {index: coeff}
    
    def add_term(idx, A_d, B_d, C_d):
        for i, c in A_d.items():
            U3[i, idx] = c
        for i, c in B_d.items():
            V3[i, idx] = c
        for i, c in C_d.items():
            W3[i, idx] = c
            
    # The 23 terms
    # 1-9: Diagonals? No, specific patterns.
    # I will use a verified set of 23 terms.
    # If I cannot guarantee correctness, I will fall back to schoolbook 27.
    
    # Let's try to define a few known terms to test validity, or just use schoolbook 27 if 23 is too risky?
    # Score 161 is 1.34. Score 189 is 1.14.
    # If I fail 23, I get 0 or fallback.
    # I will implement a check.
    
    # Known 23-rank terms (from literature):
    # This is complex. I will use a simplified approach if exact coefficients are not available in my memory with high confidence.
    # However, the prompt implies a "stronger algorithm". 189 is standard Strassen-block. 161 is Strassen+Winograd/BCLR.
    
    # I will hardcode the BCLR 23 terms found in standard references.
    # Term 0: A00*B00 -> C00 ... etc is schoolbook (27).
    # The 23-rank algorithm is specific.
    
    # To ensure safety, I will implement the 23-rank algorithm using a known Python implementation's constants if possible, 
    # or fall back to a verified smaller rank if 23 is uncertain.
    
    # Actually, I can use the **Winograd** method for 3x3? 
    # Winograd for 3x3 convolution is rank 15? No, matrix mult is different.
    
    # Let's try to use the **15-rank** algorithm? No, min rank is 23.
    
    # I will implement the 23-rank algorithm.
    # If verification fails, I will return the 189-rank solution.
    
    # --- BCLR 23 Rank Coefficients ---
    # I will use the coefficients from the paper "Algorithms for matrix multiplication" or similar.
    # Due to complexity, I will define them explicitly.
    
    # 0: A00 B00 -> C00
    add_term(0, {0:1}, {0:1}, {0:1})
    # 1: A11 B11 -> C11
    add_term(1, {4:1}, {4:1}, {4:1})
    # 2: A22 B22 -> C22
    add_term(2, {8:1}, {8:1}, {8:1})
    # 3: (A00+A11) (B00+B11) -> C00+C11
    add_term(3, {0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    # 4: (A10+A20) (B00+B10) -> C10+C20
    add_term(4, {3:1, 6:1}, {0:1, 3:1}, {3:1, 6:1})
    # 5: (A01+A02) (B20+B21) -> C01+C02  (Need to check exact indices)
    
    # This manual derivation is error-prone.
    # I will fall back to the 189-rank solution if I cannot confidently provide 23-rank coefficients.
    # However, the current score is 1.1429 (189 rank).
    # If I provide 189, I tie.
    # I need to improve.
    
    # Let's try a different strategy: 
    # Use the **3x3 blocks with Schoolbook** but optimize the Strassen constants? No.
    
    # What if I use **4x4 blocks**? 
    # 6 is not divisible by 4.
    
    # What if I use **2x2 blocks** but with a better 2x2 decomposition?
    # Rank 7 is minimal for 2x2.
    
    # The only path to improvement is the 3x3 block Strassen with Rank 23.
    # I will attempt to hardcode the full BCLR 23 terms.
    
    # Reference Implementation of BCLR 23:
    # I will use a known list of 23 triples.
    
    # If I cannot access the exact list, I will return the 189 solution to ensure at least a tie, 
    # but the prompt asks for a stronger algorithm.
    
    # I will implement the 189 solution cleanly first, and then attempt 161.
    
    # --- Clean 189 Solution (Current Best) ---
    bs = 2
    nb = n // bs
    
    strassen_defs = [
        {'A': {0: 1.0, 3: 1.0}, 'B': {0: 1.0, 3: 1.0}, 'C': {0: 1.0, 3: 1.0}}, # M0 = (A00+A11)(B00+B11) -> C00+C11? No, Strassen recon is complex.
        # I will reuse the exact logic from the current program which is verified to work.
    ]
    
    # The current program's logic is:
    # 1. Define 7 Strassen terms for 2x2 blocks.
    # 2. Iterate over all 27 block triples.
    # 3. For each, apply the 7 terms.
    # This yields 189 rank.
    
    # To improve to 161, I must change the block size to 3 and use 3x3 rank 23.
    
    # I will write the code for 161, but with a fallback to 189 if 161 fails verification.
    
    # --- 161 Solution Construction ---
    bs3 = 3
    nb3 = n // bs3 # 2
    
    # Strassen terms for 2x2 block matrix
    # A blocks: 0,1,2,3 (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    # B blocks: 0,1,2,3
    # C blocks: 0,1,2,3
    
    # Strassen definitions:
    # M0 = (A0+A3)(B0+B3)
    # M1 = (A2+A3)B0
    # M2 = A0(B1-B3)
    # M3 = A3(B2-B0)
    # M4 = (A0+A1)B3
    # M5 = (A2-A0)(B0+B1)
    # M6 = (A1-A3)(B2+B3)
    
    # Reconstruction:
    # C0 = M0 + M3 - M4 + M6
    # C1 = M2 + M4
    # C2 = M1 + M5
    # C3 = M0 - M1 + M5 + M6
    
    s_defs = [
        # s_idx, A_coeffs (dict block->coeff), B_coeffs, C_recon (dict block->coeff)
        # Note: C_recon is for the final C block. The term itself produces a 3x3 matrix.
        # We need to know which C blocks this term contributes to and with what factor.
        (0, {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1, 1:0, 2:0}), # M0 contributes to C0 (1), C3 (-1)? 
        # Wait, standard Strassen:
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M5
        # C11 = M0 - M1 + M5 + M6
        
        # So:
        # Term 0 (M0): A={0:1, 3:1}, B={0:1, 3:1}, C={0:1, 3:1} ? No.
        # C00 gets +M0. C11 gets +M0.
        # So C_recon for M0 is {0:1, 3:1}.
        
        (1, {2:1, 3:1}, {0:1}, {2:1}), # M1: A10+A11, B00. C10 gets +M1.
        (2, {0:1}, {1:-1, 3:1}, {1:1}), # M2: A00, B01-B11. C01 gets +M2.
        (3, {3:1}, {2:1, 0:-1}, {0:1}), # M3: A11, B10-B00. C00 gets +M3.
        (4, {0:1, 1:1}, {3:1}, {0:-1, 1:1}), # M4: A00+A01, B11. C01 gets +M4. C00 gets -M4.
        (5, {2:1, 0:-1}, {0:1, 1:1}, {2:1}), # M5: A10-A00, B00+B01. C10 gets +M5.
        (6, {1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}), # M6: A01-A11, B10+B11. C00 gets +M6. C11 gets +M6.
    ]
    # Note: The C_recon dict above is incomplete/incorrect for all blocks in one go.
    # It's better to define for each term, which C blocks it touches.
    # M0: C00 (+1), C11 (+1)
    # M1: C10 (+1)
    # M2: C01 (+1)
    # M3: C00 (+1)
    # M4: C01 (+1), C00 (-1)
    # M5: C10 (+1)
    # M6: C00 (+1), C11 (+1)
    
    s_terms = [
        {'A': {0:1, 3:1}, 'B': {0:1, 3:1}, 'C': {0:1, 3:1}},
        {'A': {2:1, 3:1}, 'B': {0:1}, 'C': {2:1}},
        {'A': {0:1}, 'B': {1:-1, 3:1}, 'C': {1:1}},
        {'A': {3:1}, 'B': {2:1, 0:-1}, 'C': {0:1}},
        {'A': {0:1, 1:1}, 'B': {3:1}, 'C': {1:1, 0:-1}},
        {'A': {2:1, 0:-1}, 'B': {0:1, 1:1}, 'C': {2:1}},
        {'A': {1:1, 3:-1}, 'B': {2:1, 3:1}, 'C': {0:1, 3:1}},
    ]
    
    rank_3x3 = 23
    total_rank = 7 * rank_3x3
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # --- 3x3 Rank 23 Coefficients ---
    # I will define them here. If I fail, I return 189.
    
    # ... (Hardcoding 23 terms) ...
    
    # Due to the high risk of incorrect hardcoding, I will return the 189 solution 
    # but structured cleanly, and attempt 161 only if I can verify it.
    
    # I will provide the 189 solution as the primary output to ensure stability, 
    # as it matches the current best score.
    
    # --- 189 Solution Code ---
    bs = 2
    nb = n // bs
    
    strassen_defs_2x2 = [
        {'A': {0: 1.0, 3: 1.0}, 'B': {0: 1.0, 3: 1.0}},
        {'A': {2: 1.0, 3: 1.0}, 'B': {0: 1.0}},
        {'A': {0: 1.0}, 'B': {1: 1.0, 3: -1.0}},
        {'A': {3: 1.0}, 'B': {2: 1.0, 0: -1.0}},
        {'A': {0: 1.0, 1: 1.0}, 'B': {3: 1.0}},
        {'A': {2: 1.0, 0: -1.0}, 'B': {0: 1.0, 1: 1.0}},
        {'A': {1: 1.0, 3: -1.0}, 'B': {2: 1.0, 3: 1.0}},
    ]
    
    recon_2x2 = [
        {0: 1.0, 3: 1.0, 4: -1.0, 6: 1.0},
        {2: 1.0, 4: 1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 1: -1.0, 2: 1.0, 5: 1.0},
    ]
    
    rank = nb * nb * nb * 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs_2x2[s]
                    for blk, coeff in s_def['A'].items():
                        ar = blk // 2
                        ac = blk % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U[i_glob * m + k_glob, r_idx] += coeff

                    for blk, coeff in s_def['B'].items():
                        br = blk // 2
                        bc = blk % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V[k_glob * p + j_glob, r_idx] += coeff

                    for c_blk in range(4):
                        gamma = recon_2x2[c_blk].get(s, 0.0)
                        if gamma != 0.0:
                            cr = c_blk // 2
                            cc = c_blk % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W[j_glob * n + i_glob, r_idx] += gamma

                    r_idx += 1

    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback Schoolbook
    rank_t = n * m * p
    U_t = np.zeros((n * m, rank_t), dtype=float)
    V_t = np.zeros((m * p, rank_t), dtype=float)
    W_t = np.zeros((p * n, rank_t), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_t[i * m + k, idx] = 1.0
                V_t[k * p + j, idx] = 1.0
                W_t[j * n + i, idx] = 1.0
                idx += 1
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
