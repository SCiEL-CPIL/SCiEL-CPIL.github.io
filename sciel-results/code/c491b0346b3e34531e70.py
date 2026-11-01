"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 34 decomposition.
    # Split B into B1 (3x2) and B2 (3x5).
    # Split A into A1 (2x2) and A2 (2x1).
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x2x2 (Rank 7 via Strassen)
    # A1*B2 is 2x2x5 (Rank 20 via Schoolbook)
    # A2*B2 is 2x1x7 (Rank 7 via Schoolbook)
    # Total Rank = 7 + 20 + 7 = 34
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)] # 6 x 34
    V = [[0] * rank for _ in range(m * p)] # 21 x 34
    W = [[0] * rank for _ in range(p * n)] # 14 x 34
    
    col = 0
    
    # 1. Strassen for A1 (2x2) * B1 (2x2 part of B, which is B[0:2, 0:2])
    # A1 indices: U[0:2, :], V[0:2, :] (rows 0,1 of B)
    # B1 indices: V[0:2, :], W[0:2, :] (cols 0,1 of C)
    
    # Strassen 7 terms for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We map these to the global indices.
    # A1 is rows 0,1 of A (global U indices 0,1, 2,3? No, A is 2x3.
    # A1 is A[:, 0:2]. So U indices: 0,1 (col 0), 2,3 (col 1).
    # Wait, U shape is (n*m, rank) = (6, rank).
    # U[i*m + j, c].
    # A1 corresponds to i=0,1 and j=0,1.
    # U[0, c] = a00, U[1, c] = a10, U[2, c] = a01, U[3, c] = a11.
    
    # B1 corresponds to j=0,1 and k=0,1.
    # V[j*p + k, c].
    # V[0, c] = b00, V[1, c] = b10, V[3, c] = b01, V[4, c] = b11.
    # Note: j=0, k=0 -> 0. j=0, k=1 -> 1. j=1, k=0 -> 3. j=1, k=1 -> 4.
    
    # C1 corresponds to k=0,1 and i=0,1.
    # W[k*n + i, c].
    # W[0, c] = c00, W[1, c] = c10, W[2, c] = c01, W[3, c] = c11.
    
    # Term 1: (a00+a11)(b00+b11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[3][col] = 1; W[1][col] = 1; W[2][col] = 1 # Wait, C = (A11+A22)(B11+B22) contributes to all 4?
    # C11 = A11B11 + A12B21 + ...
    # Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's implement the 7 terms correctly.
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx][col] += val
        for idx, val in v_vals.items():
            V[idx][col] += val
        for idx, val in w_vals.items():
            W[idx][col] += val
        col += 1

    # Strassen Terms
    # M1 = (A00+A11)(B00+B11)
    add_term({0: 1, 3: 1}, {0: 1, 4: 1}, {0: 1, 1: 1, 2: 1, 3: 1}) # This is wrong. W is output.
    # The term is u * v * w.
    # u is vector in A space, v in B space, w in C space.
    # C += u v^T w^T? No.
    # C_{ik} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # So for a rank-1 term, U is a vector, V is a vector, W is a vector.
    # U[i*m+j] is coeff for A_{i,j}.
    # V[j*p+k] is coeff for B_{j,k}.
    # W[k*n+i] is coeff for C_{k,i}.
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (0,0) -> idx 0. A11 (1,1) -> idx 3.
    # V: B00 (0,0) -> idx 0. B11 (1,1) -> idx 4.
    # W: This term contributes to C00, C01, C10, C11?
    # No, M1 is a scalar. It is added to specific C elements.
    # C00 += M1. C11 += M1. C01 -= M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # U: {0: 1, 3: 1}
    # V: {0: 1, 4: 1}
    # W: {0: 1, 3: 1} # C00 (k=0,i=0) and C11 (k=1,i=1)
    # Wait, W index is k*n+i.
    # C00: k=0, i=0 -> 0.
    # C11: k=1, i=1 -> 1*2+1 = 3.
    # So W: {0: 1, 3: 1}.
    
    col = 0
    # M1
    add_term({0: 1, 3: 1}, {0: 1, 4: 1}, {0: 1, 3: 1})
    # M2 = (A10+A11)B00
    add_term({2: 1, 3: 1}, {0: 1}, {1: 1}) # C10 (k=0,i=1) -> idx 1
    # M3 = A00(B10-B00)
    add_term({0: 1}, {3: 1, 0: -1}, {2: 1}) # C01 (k=1,i=0) -> idx 2
    # M4 = A11(B01-B11)
    add_term({3: 1}, {1: 1, 4: -1}, {1: 1, 0: 1}) # C10 (1) and C00 (0)
    # M5 = (A00+A01)B11
    add_term({0: 1, 1: 1}, {4: 1}, {2: 1, 0: -1}) # C01 (2) and C00 (0)
    # M6 = (A10-A00)(B00+B01)
    add_term({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}) # C11 (3)
    # M7 = (A01-A11)(B10+B11)
    add_term({1: 1, 3: -1}, {3: 1, 4: 1}, {0: 1, 2: -1}) # C00 (0) and C01 (2)
    
    # 2. Schoolbook for A1 (2x2) * B2 (2x5)
    # B2 is B[0:2, 2:7]
    # j=0,1. k=2..6.
    # 2*2*5 = 20 terms.
    for i in range(2):
        for j in range(2):
            for k in range(2, 7):
                u_idx = i * 3 + j
                v_idx = j * 7 + k
                w_idx = k * 2 + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                
    # 3. Schoolbook for A2 (2x1) * B2 (1x7)
    # A2 is A[:, 2]. i=0,1. j=2.
    # B2 is B[2, 0:7]. j=2. k=0..6.
    # 2*1*7 = 14 terms?
    # Wait, my previous split was:
    # C = A1*B1 + A1*B2 + A2*B2
    # A1 is 2x2, B1 is 2x2 (cols 0,1 of B).
    # A1 is 2x2, B2 is 2x5 (cols 2-6 of B).
    # A2 is 2x1 (col 2 of A), B2 is 1x7 (row 2 of B).
    
    # I already did A1*B1 (7 terms) and A1*B2 (20 terms).
    # Now A2*B2.
    # A2: i=0,1; j=2.
    # B2: j=2; k=0..6.
    for i in range(2):
        for k in range(7):
            u_idx = i * 3 + 2
            v_idx = 2 * 7 + k
            w_idx = k * 2 + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
    # Total rank = 7 + 20 + 14 = 41?
    # My previous calculation:
    # A1*B1: 7
    # A1*B2: 20
    # A2*B2: 14
    # Total 41.
    # This is barely better than 42.
    
    # Let's re-evaluate.
    # A is 2x3. B is 3x7.
    # Split B into B_top (2x7) and B_bottom (1x7).
    # C = A_top * B_top + A_bottom * B_bottom.
    # A_top is 2x2. A_bottom is 2x1.
    # A_top * B_top is 2x2x7. Rank 14 (schoolbook) or 7*? (Strassen on 2x2x2 blocks).
    # If we use Strassen for 2x2x2 (7 terms) for the first 2 cols of B_top, and schoolbook for the rest (5 cols, 20 terms).
    # 7 + 20 = 27.
    # A_bottom * B_bottom is 2x1x7. Rank 14.
    # Total 27 + 14 = 41.
    
    # Can we do better?
    # 2x2x7 rank is 14? No, 2x2x2 is 7. 2x2x4 is 14? 2x2x7 is 21?
    # Strassen for 2x2x4: Two 2x2x2 blocks. 7+7=14.
    # 2x2x7: 2x2x4 (14) + 2x2x3 (10?) = 24.
    # Total 24 + 14 = 38.
    
    # I will stick with Rank 41 for now as it is safely better than 42 and easy to verify.
    # Actually, 41 is very close to 42.
    
    # Let's try to optimize the 2x2x7 part.
    # 2x2x7 can be done with rank 14? No.
    # Rank of 2x2xN is 2N for N>=2? No, Strassen reduces it.
    # Rank of 2x2x2 is 7.
    # Rank of 2x2x3 is 10.
    # Rank of 2x2x4 is 14.
    # Rank of 2x2x7 is 7 (for 2) + 7 (for 2) + 10 (for 3) = 24.
    
    # So 2x2x7 rank 24.
    # 2x1x7 rank 14.
    # Total 38.
    
    # I will implement Rank 38.
    
    # Reset
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 1. A_top (2x2) * B_top (2x7)
    # Split B_top into B_t1 (2x2), B_t2 (2x2), B_t3 (2x3).
    # B_t1: cols 0,1.
    # B_t2: cols 2,3.
    # B_t3: cols 4,5,6.
    
    # 1.1 A_top * B_t1 (2x2x2) -> Rank 7 (Strassen)
    # Same as before, but only for cols 0,1 of B.
    # U indices: 0,1,2,3.
    # V indices: 0,1,3,4 (cols 0,1).
    # W indices: 0,1,2,3 (cols 0,1).
    
    add_term({0: 1, 3: 1}, {0: 1, 4: 1}, {0: 1, 3: 1})
    add_term({2: 1, 3: 1}, {0: 1}, {1: 1})
    add_term({0: 1}, {3: 1, 0: -1}, {2: 1})
    add_term({3: 1}, {1: 1, 4: -1}, {1: 1, 0: 1})
    add_term({0: 1, 1: 1}, {4: 1}, {2: 1, 0: -1})
    add_term({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_term({1: 1, 3: -1}, {3: 1, 4: 1}, {0: 1, 2: -1})
    
    # 1.2 A_top * B_t2 (2x2x2) -> Rank 7 (Strassen)
    # B_t2 cols 2,3.
    # V indices: j=0,1. k=2,3.
    # V[0*7+2]=2, V[0*7+3]=3, V[1*7+2]=9, V[1*7+3]=10.
    # W indices: k=2,3. i=0,1.
    # W[2*2+0]=4, W[2*2+1]=5, W[3*2+0]=6, W[3*2+1]=7.
    
    add_term({0: 1, 3: 1}, {2: 1, 10: 1}, {4: 1, 7: 1})
    add_term({2: 1, 3: 1}, {2: 1}, {5: 1})
    add_term({0: 1}, {9: 1, 2: -1}, {6: 1})
    add_term({3: 1}, {3: 1, 10: -1}, {5: 1, 4: 1})
    add_term({0: 1, 1: 1}, {10: 1}, {6: 1, 4: -1})
    add_term({2: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add_term({1: 1, 3: -1}, {9: 1, 10: 1}, {4: 1, 6: -1})
    
    # 1.3 A_top * B_t3 (2x2x3) -> Rank 10
    # B_t3 cols 4,5,6.
    # We can do 2x2x2 (cols 4,5) + 2x2x1 (col 6).
    # 2x2x2 Rank 7.
    # 2x2x1 Rank 4.
    # Total 11.
    # Or use known 2x2x3 rank 10.
    
    # Let's do 2x2x2 (cols 4,5) Rank 7.
    # V indices: k=4,5.
    # V[4]=4, V[5]=5, V[11]=11, V[12]=12.
    # W indices: k=4,5.
    # W[8]=8, W[9]=9, W[10]=10, W[11]=11.
    
    add_term({0: 1, 3: 1}, {4: 1, 12: 1}, {8: 1, 11: 1})
    add_term({2: 1, 3: 1}, {4: 1}, {9: 1})
    add_term({0: 1}, {11: 1, 4: -1}, {10: 1})
    add_term({3: 1}, {5: 1, 12: -1}, {9: 1, 8: 1})
    add_term({0: 1, 1: 1}, {12: 1}, {10: 1, 8: -1})
    add_term({2: 1, 0: -1}, {4: 1, 5: 1}, {11: 1})
    add_term({1: 1, 3: -1}, {11: 1, 12: 1}, {8: 1, 10: -1})
    
    # 2x2x1 (col 6) Rank 4.
    # V index: k=6. V[6]=6, V[13]=13.
    # W index: k=6. W[12]=12, W[13]=13.
    add_term({0: 1}, {6: 1}, {12: 1})
    add_term({1: 1}, {6: 1}, {13: 1})
    add_term({2: 1}, {13: 1}, {12: 1})
    add_term({3: 1}, {13: 1}, {13: 1})
    
    # 2. A_bottom (2x1) * B_bottom (1x7) -> Rank 14
    # A_bottom: j=2.
    # B_bottom: j=2.
    for i in range(2):
        for k in range(7):
            u_idx = i * 3 + 2
            v_idx = 2 * 7 + k
            w_idx = k * 2 + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
