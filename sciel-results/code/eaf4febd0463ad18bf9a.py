def construct():
    n, m, p = 2, 3, 6
    
    # Rank 17 decomposition for 2x3x3
    # 7 Strassen terms + 4 (A_1:2,3 * B_3,1:2) + 4 (A_1:2,1:2 * B_1:2,3) + 2 (A_1:2,3 * B_3,3)
    # Wait, I combined A_1:2,3 * B_3,1:3 into rank 6.
    # And A_1:2,1:2 * B_1:2,3 is rank 4.
    # Total 7 + 6 + 4 = 17.
    
    rank_per_block = 17
    total_rank = rank_per_block * 2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_block(offset, block_offset):
        # offset: column offset in U,V,W
        # block_offset: row offset in V (0 or 3)
        
        # Strassen terms 0-6
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Helper to add a term
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
            col = offset
            U[u_idx][col] += u_val
            V[v_idx + block_offset][col] += v_val
            W[w_idx][col] += w_val
            
        # M1
        add_term(0, 1, 0, 1, 0, 1) # A00 B00 -> C00 (partially)
        add_term(3, 1, 0, 1, 0, 1) # A11 B00
        add_term(0, 1, 4, 1, 0, 1) # A00 B11
        add_term(3, 1, 4, 1, 0, 1) # A11 B11
        # Wait, Strassen terms are products of sums.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is 4 terms if expanded, but Strassen counts it as 1 "multiplication" of the sums.
        # In tensor decomposition, we need to represent the linear combinations.
        # u vector has 1 at A00 and A11.
        # v vector has 1 at B00 and B11.
        # w vector gets the contribution to C.
        
        # Let's redefine add_term to take vectors.
        def add_vec_term(u_vec, v_vec, w_vec):
            col = offset
            for idx, val in u_vec.items():
                U[idx][col] += val
            for idx, val in v_vec.items():
                V[idx + block_offset][col] += val
            for idx, val in w_vec.items():
                W[idx][col] += val

        # M1: (A00+A11) * (B00+B11)
        # Contributes to C00, C10? No, C00, C01, C10, C11?
        # C00 += M1
        # C01 += M1? No.
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        add_vec_term({0: 1, 3: 1}, {0: 1, 4: 1}, {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}) # W indices: C00=0, C01=1, C10=2, C11=3?
        # W index: k*n + i.
        # C00: k=0, i=0 -> 0.
        # C01: k=0, i=1 -> 1.
        # C10: k=1, i=0 -> 2.
        # C11: k=1, i=1 -> 3.
        # So W indices 0,1,2,3.
        
        # I need to be careful with W indices.
        # M1 contributes to C00 and C11?
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So W[0] += 1, W[3] += 1.
        
        add_vec_term({0: 1, 3: 1}, {0: 1, 4: 1}, {0: 1, 3: 1})
        
        # M2: (A10+A11) * B00
        # A10 is idx 1, A11 is idx 3.
        # B00 is idx 0.
        # Contributes to C10 (idx 2) and C11 (idx 3)?
        # C10 = M2 + ...
        # C11 = ... - M2 ...
        add_vec_term({1: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        
        # M3: A00 * (B01-B11)
        # A00 idx 0.
        # B01 idx 1, B11 idx 4.
        # Contributes to C01 (idx 1) and C11 (idx 3).
        add_vec_term({0: 1}, {1: 1, 4: -1}, {1: 1, 3: 1})
        
        # M4: A11 * (B10-B00)
        # A11 idx 3.
        # B10 idx 3, B00 idx 0.
        # Contributes to C00 (idx 0) and C10 (idx 2).
        add_vec_term({3: 1}, {3: 1, 0: -1}, {0: 1, 2: 1})
        
        # M5: (A00+A01) * B11
        # A00 idx 0, A01 idx 2.
        # B11 idx 4.
        # Contributes to C00 (idx 0) and C01 (idx 1).
        add_vec_term({0: 1, 2: 1}, {4: 1}, {0: -1, 1: 1})
        
        # M6: (A10-A00) * (B00+B01)
        # A10 idx 1, A00 idx 0.
        # B00 idx 0, B01 idx 1.
        # Contributes to C11 (idx 3).
        add_vec_term({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        
        # M7: (A01-A11) * (B10+B11)
        # A01 idx 2, A11 idx 3.
        # B10 idx 3, B11 idx 4.
        # Contributes to C00 (idx 0).
        add_vec_term({2: 1, 3: -1}, {3: 1, 4: 1}, {0: 1})

        # Now the extra terms for 2x3x3
        # A_1:2,3 is A02 (idx 2) and A12 (idx 5).
        # B_3,1:2 is B20 (idx 6) and B21 (idx 7).
        # B_3,3 is B22 (idx 8).
        # B_1:2,3 is B02 (idx 2) and B12 (idx 5).
        
        # Term Group 2: A_1:2,3 * B_3,1:3 (Rank 6)
        # A02 * B20 -> C00? No, C_{0,0} = A00B00 + A01B10 + A02B20.
        # So A02 * B20 contributes to C00 (idx 0).
        add_vec_term({2: 1}, {6: 1}, {0: 1})
        
        # A02 * B21 -> C01 (idx 1).
        add_vec_term({2: 1}, {7: 1}, {1: 1})
        
        # A02 * B22 -> C02 (idx 4? k=2, i=0 -> 2*2+0=4).
        add_vec_term({2: 1}, {8: 1}, {4: 1})
        
        # A12 * B20 -> C10 (idx 2).
        add_vec_term({5: 1}, {6: 1}, {2: 1})
        
        # A12 * B21 -> C11 (idx 3).
        add_vec_term({5: 1}, {7: 1}, {3: 1})
        
        # A12 * B22 -> C12 (idx 5).
        add_vec_term({5: 1}, {8: 1}, {5: 1})

        # Term Group 3: A_1:2,1:2 * B_1:2,3 (Rank 4)
        # A00 * B02 -> C02 (idx 4).
        add_vec_term({0: 1}, {2: 1}, {4: 1})
        
        # A01 * B12 -> C02 (idx 4).
        add_vec_term({2: 1}, {5: 1}, {4: 1})
        
        # A10 * B02 -> C12 (idx 5).
        add_vec_term({1: 1}, {2: 1}, {5: 1})
        
        # A11 * B12 -> C12 (idx 5).
        add_vec_term({3: 1}, {5: 1}, {5: 1})

    # Block 0
    add_block(0, 0)
    
    # Block 1
    # V offset is 3 (rows 3,4,5 for B)
    # W offset? W is shared? No, W is for C.
    # C is 2x6.
    # Block 0 C cols 0,1,2.
    # Block 1 C cols 3,4,5.
    # W index k*n + i.
    # k is column of C.
    # Block 0: k=0,1,2.
    # Block 1: k=3,4,5.
    # So W indices for Block 1 should be offset by 3*2 = 6?
    # W shape is (p*n, rank) = (12, rank).
    # Block 0 uses W[0:6].
    # Block 1 uses W[6:12].
    
    # I need to modify add_block to accept W offset.
    
    def add_block_v2(offset, v_offset, w_offset):
        def add_vec_term(u_vec, v_vec, w_vec):
            col = offset
            for idx, val in u_vec.items():
                U[idx][col] += val
            for idx, val in v_vec.items():
                V[idx + v_offset][col] += val
            for idx, val in w_vec.items():
                W[idx + w_offset][col] += val

        # M1
        add_vec_term({0: 1, 3: 1}, {0: 1, 4: 1}, {0: 1, 3: 1})
        # M2
        add_vec_term({1: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        # M3
        add_vec_term({0: 1}, {1: 1, 4: -1}, {1: 1, 3: 1})
        # M4
        add_vec_term({3: 1}, {3: 1, 0: -1}, {0: 1, 2: 1})
        # M5
        add_vec_term({0: 1, 2: 1}, {4: 1}, {0: -1, 1: 1})
        # M6
        add_vec_term({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        # M7
        add_vec_term({2: 1, 3: -1}, {3: 1, 4: 1}, {0: 1})

        # Extra 1
        add_vec_term({2: 1}, {6: 1}, {0: 1})
        add_vec_term({2: 1}, {7: 1}, {1: 1})
        add_vec_term({2: 1}, {8: 1}, {4: 1})
        add_vec_term({5: 1}, {6: 1}, {2: 1})
        add_vec_term({5: 1}, {7: 1}, {3: 1})
        add_vec_term({5: 1}, {8: 1}, {5: 1})

        # Extra 2
        add_vec_term({0: 1}, {2: 1}, {4: 1})
        add_vec_term({2: 1}, {5: 1}, {4: 1})
        add_vec_term({1: 1}, {2: 1}, {5: 1})
        add_vec_term({3: 1}, {5: 1}, {5: 1})

    add_block_v2(0, 0, 0)
    add_block_v2(17, 3, 6)

    return {"U": U, "V": V, "W": W}
