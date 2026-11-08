"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # New approach: direct rank-24 construction using column-pair Strassen
    # Group 6 columns into 3 pairs, use rank-7 for each 2x3x2 subproblem
    # Total rank = 3 * 7 = 21
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]   # 6 x 21
    V = [[0] * rank for _ in range(m * p)]   # 18 x 21
    W = [[0] * rank for _ in range(p * n)]   # 12 x 21
    
    # For each column pair b in {0,1,2}, handle columns 2b and 2b+1
    # 2x3x2 decomposition in rank 7:
    # A: 2x3 (indices 0-5: A00,A01,A02,A10,A11,A12)
    # B: 3x2 (indices 0-5: B00,B01,B10,B11,B20,B21)
    # C: 2x2 (indices 0-3: C00,C01,C10,C11)
    
    # Strassen-style 2x3x2 in rank 7:
    # M1 = (A00+A12)*(B00+B10+B20) -> contributes to C00, C10
    # M2 = (A01+A12)*(B01+B11+B21) -> contributes to C01, C11
    # M3 = (A00-A10)*(B00+B10) -> contributes to C00, C10
    # M4 = (A01-A11)*(B01+B11) -> contributes to C01, C11
    # M5 = A02*(B20+B21) -> contributes to C00, C01
    # M6 = A12*(B20-B21) -> contributes to C10, C11
    # M7 = (A00+A01+A10+A11)*(B00+B01+B10+B11) -> correction terms
    
    # Actually, let me use a verified 2x3x2 rank-7 construction:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    
    # Rank-7 construction:
    # P1 = (A00+A10) * (B00+B01)
    # P2 = (A01+A11) * (B10+B11)
    # P3 = (A02+A12) * (B20+B21)
    # P4 = (A00-A10) * B00
    # P5 = (A01-A11) * B10
    # P6 = (A02-A12) * B20
    # P7 = (A00+A01+A02) * (B01+B11+B21)
    
    # Verify C00: need A00*B00 + A01*B10 + A02*B20
    # From P1: A00*B00 + A00*B01 + A10*B00 + A10*B01
    # From P2: A01*B10 + A01*B11 + A11*B10 + A11*B11
    # From P3: A02*B20 + A02*B21 + A12*B20 + A12*B21
    # From P4: A00*B00 - A10*B00
    # From P5: A01*B10 - A11*B10
    # From P6: A02*B20 - A12*B20
    # From P7: A00*B01 + A00*B11 + A00*B21 + A01*B01 + A01*B11 + A01*B21 + A02*B01 + A02*B11 + A02*B21
    # Sum for C00: A00*B00(1+1) + A01*B10(1+1) + A02*B20(1+1) + ... this is getting messy.
    
    # Let me use a known valid 2x3x2 rank-7:
    # Use the standard approach: split into 2x2x2 (rank 7) + 2x1x2 (rank 2) - 2x2x1 (rank 2) = rank 7
    # Or just use 11 terms (simpler) for reliability.
    
    # For safety, use rank-11 per block, total rank 33. But let me try rank-8 per block = 24 total.
    
    # Actually, let me use a clean rank-24 construction:
    # For each j in {0,1,2}, compute 8 terms to handle all 6 k-values for row 0 and row 1.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each j, use 8 terms:
    # Terms for j-th column of A (2 elements) times 6 columns of B row j
    # A[j_col] = [a0, a1], B[j_row] = [b0, b1, b2, b3, b4, b5]
    # Need: c0k = a0*bk, c1k = a1*bk for k=0..5
    
    # 8-term construction per j:
    # T1 = (a0+a1)*b0 -> c00 += T1, c10 += T1
    # T2 = (a0-a1)*b0 -> c00 += T2, c10 -= T2
    # T3 = (a0+a1)*b1 -> c01 += T3, c11 += T3
    # T4 = (a0-a1)*b1 -> c01 += T4, c11 -= T4
    # T5 = (a0+a1)*b2 -> c02 += T5, c12 += T5
    # T6 = (a0-a1)*b2 -> c02 += T6, c12 -= T6
    # T7 = (a0+a1)*b3 -> c03 += T7, c13 += T7
    # T8 = (a0-a1)*b3 -> c03 += T8, c13 -= T8
    # This only handles 4 columns. Need more.
    
    # For 6 columns, use 12 terms per j (no savings) or find a better grouping.
    
    # Let me just use the working rank-33 and verify it, then see if I can reduce.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x3x2 block decomposition (rank 11 per block)
    block_terms = []
    
    # Strassen 2x2x2 terms (for A[:,0:2] * B[0:2,:])
    block_terms.append(({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    block_terms.append(({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}))
    block_terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
    block_terms.append(({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
    block_terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
    block_terms.append(({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
    block_terms.append(({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}))
    
    # Scalar terms for A[:,2] * B[2,:]
    block_terms.append(({2: 1}, {4: 1}, {0: 1}))
    block_terms.append(({2: 1}, {5: 1}, {1: 1}))
    block_terms.append(({5: 1}, {4: 1}, {2: 1}))
    block_terms.append(({5: 1}, {5: 1}, {3: 1}))
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(block_terms):
            col = block * 11 + t_idx
            
            for idx, val in u_dict.items():
                U[idx][col] = val
            
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
