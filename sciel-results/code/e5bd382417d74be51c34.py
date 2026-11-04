"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strategy:
    # Split n=3 into [1, 2]
    # Split m=4 into [3, 1]
    # p=8 stays whole
    #
    # Blocks:
    # 1. (1,3,8): rank 24 (schoolbook)
    # 2. (1,1,8): rank 8 (schoolbook)
    # 3. (2,3,8): rank 28 (Laderman 2x3x4 applied twice for p=4,4)
    # 4. (2,1,8): rank 16 (schoolbook)
    # Total: 24 + 8 + 28 + 16 = 76
    
    # Actually, let me try to get a better decomposition for 2x3x8.
    # 2x3x8: We can split p=8 into [4,4] and use Laderman (rank 14) for each 2x3x4.
    # So rank(2x3x8) = 2 * 14 = 28.
    
    # Let me implement Laderman's 2x3x4 algorithm properly.
    
    # Laderman's 2x3x4 algorithm (rank 14):
    # A is 2x3, B is 3x4, C is 2x4
    # A[i,j] for i in {0,1}, j in {0,1,2}
    # B[i,j] for i in {0,1,2}, j in {0,1,2,3}
    # C[i,j] for i in {0,1}, j in {0,1,2,3}
    
    # I'll use a known rank-14 construction.
    
    # Let me define the 14 bilinear forms.
    # Each form is (u_coeffs, v_coeffs, w_coeffs) where:
    # u_coeffs: dict from A index (i*3+j) to coefficient
    # v_coeffs: dict from B index (i*4+j) to coefficient
    # w_coeffs: dict from C index (i*4+j) to coefficient
    
    # I'll use a construction based on the following idea:
    # Split A into columns: A = [A0, A1, A2] where Ai is 2x1
    # Split B into rows: B = [B0; B1; B2] where Bi is 1x4
    # Split C into columns: C = [C0, C1, C2, C3] where Ci is 2x1
    
    # The product is: C[:,k] = A0*B0[k] + A1*B1[k] + A2*B2[k] for k=0,1,2,3
    
    # This is a sum of 3 rank-4 tensors (one for each column of A).
    # Each rank-4 tensor has rank 4 (schoolbook for 2x1x4).
    # So naive rank is 3*4 = 12... wait, that's already less than 14.
    
    # Hmm, let me reconsider. Actually for 2x1x4, the rank is 4 (schoolbook).
    # So 3 such blocks give rank 12. But we need to handle the fact that
    # the 3 blocks share the same A and C structure.
    
    # Actually, I think the minimal rank for 2x3x4 is indeed 14, not 12.
    # Let me look up or derive a proper rank-14 construction.
    
    # Let me try a different approach. I'll use the following known result:
    # The tensor rank of 2x3x4 is 14.
    
    # Here's a construction:
    # Let A = [[a1, a2, a3], [b1, b2, b3]]
    # Let B = [[c1, c2, c3, c4], [d1, d2, d3, d4], [e1, e2, e3, e4]]
    # Let C = [[f1, f2, f3, f4], [g1, g2, g3, g4]]
    
    # where C[i,j] = sum_k A[i,k] * B[k,j]
    
    # I'll define 14 products:
    
    laderman_2x3x4_terms = []
    
    # I'll use a construction from the literature.
    # Let me define the 14 terms as (u_dict, v_dict, w_dict).
    
    # Term 1: a1*c1 -> f1
    laderman_2x3x4_terms.append(({0: 1}, {0: 1}, {0: 1}))
    # Term 2: a1*c2 -> f2
    laderman_2x3x4_terms.append(({0: 1}, {1: 1}, {1: 1}))
    # Term 3: a1*c3 -> f3
    laderman_2x3x4_terms.append(({0: 1}, {2: 1}, {2: 1}))
    # Term 4: a1*c4 -> f4
    laderman_2x3x4_terms.append(({0: 1}, {3: 1}, {3: 1}))
    # Term 5: b1*d1 -> g1
    laderman_2x3x4_terms.append(({3: 1}, {4: 1}, {4: 1}))
    # Term 6: b1*d2 -> g2
    laderman_2x3x4_terms.append(({3: 1}, {5: 1}, {5: 1}))
    # Term 7: b1*d3 -> g3
    laderman_2x3x4_terms.append(({3: 1}, {6: 1}, {6: 1}))
    # Term 8: b1*d4 -> g4
    laderman_2x3x4_terms.append(({3: 1}, {7: 1}, {7: 1}))
    # Term 9: a2*d1 -> f1
    laderman_2x3x4_terms.append(({1: 1}, {4: 1}, {0: 1}))
    # Term 10: a2*d2 -> f2
    laderman_2x3x4_terms.append(({1: 1}, {5: 1}, {1: 1}))
    # Term 11: a2*d3 -> f3
    laderman_2x3x4_terms.append(({1: 1}, {6: 1}, {2: 1}))
    # Term 12: a2*d4 -> f4
    laderman_2x3x4_terms.append(({1: 1}, {7: 1}, {3: 1}))
    # Term 13: b2*e1 -> g1
    laderman_2x3x4_terms.append(({4: 1}, {8: 1}, {4: 1}))
    # Term 14: b2*e2 -> g2
    laderman_2x3x4_terms.append(({4: 1}, {9: 1}, {5: 1}))
    
    # Wait, this only gives 14 terms but I'm missing contributions from a3, b3, and e3,e4.
    # This is not a valid decomposition.
    
    # Let me restart with a proper approach.
    # I'll just use schoolbook for 2x3x4 (rank 24) for now, which gives total rank:
    # 24 + 8 + 24 + 8 + 24 + 8 + 24 + 8 = 128
    # That's worse than the current 76.
    
    # Let me keep the current approach but try to optimize the 2x3x4 blocks.
    
    # Actually, the current program already uses schoolbook for everything, giving:
    # Block 1: 1*3*4 = 12
    # Block 2: 1*1*4 = 4
    # Block 3: 1*3*4 = 12
    # Block 4: 1*1*4 = 4
    # Block 5: 2*3*4 = 24
    # Block 6: 2*1*4 = 8
    # Block 7: 2*3*4 = 24
    # Block 8: 2*1*4 = 8
    # Total: 96
    
    # But the current program says rank 76. Let me check...
    # Oh wait, the current program uses Laderman (14) for blocks 5 and 7, but the
    # actual implementation falls back to schoolbook. So the actual rank is 96.
    
    # Let me implement a proper rank-14 decomposition for 2x3x4.
    
    # I found a construction. Here it is:
    
    # A = [[a, b, c], [d, e, f]]
    # B = [[g, h, i, j], [k, l, m, n], [o, p, q, r]]
    # C = [[s, t, u, v], [w, x, y, z]]
    
    # s = a*g + b*k + c*o
    # t = a*h + b*l + c*p
    # u = a*i + b*m + c*q
    # v = a*j + b*n + c*r
    # w = d*g + e*k + f*o
    # x = d*h + e*l + f*p
    # y = d*i + e*m + f*q
    # z = d*j + e*n + f*r
    
    # I'll use the following 14 products:
    # P1 = a*g
    # P2 = a*h
    # P3 = a*i
    # P4 = a*j
    # P5 = d*k
    # P6 = d*l
    # P7 = d*m
    # P8 = d*n
    # P9 = (b+e)*(k+l)  -- no, this doesn't work cleanly.
    
    # Let me try a different construction. I'll use the fact that 2x3x4 can be
    # decomposed using 14 multiplications by splitting it into two 2x2x2 blocks
    # and one 2x1x4 block, but that gives 7+7+4 = 18, not 14.
    
    # I think I need to look up the actual Laderman algorithm. Since I can't,
    # let me just use schoolbook and accept rank 96.
    
    # Actually, let me try a different partitioning strategy that might give
    # a lower rank.
    
    # Alternative: Split m=4 into [2,2]
    # Blocks:
    # 1. (3,2,8): rank 48 (schoolbook)
    # 2. (3,2,8): rank 48 (schoolbook)
    # Total: 96. Same.
    
    # Let me try splitting n=3 into [2,1] and m=4 into [2,2]:
    # 1. (2,2,8): rank 32
    # 2. (2,2,8): rank 32
    # 3. (1,2,8): rank 16
    # 4. (1,2,8): rank 16
    # Total: 96. Same.
    
    # It seems like schoolbook always gives 96 for 3x4x8.
    
    # Let me just implement the schoolbook decomposition cleanly.
    
    columns = []
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    # Schoolbook for the full 3x4x8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
