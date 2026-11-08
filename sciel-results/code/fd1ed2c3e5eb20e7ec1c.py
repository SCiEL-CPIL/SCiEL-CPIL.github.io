import numpy as np

def construct():
    n, m, p = 3, 4, 4
    # Block-diagonal decomposition: 3x4x4 = 1x4x4 (row 0) + 2x4x4 (rows 1,2)
    # 1x4x4 rank 4, 2x4x4 rank 12 (via four 2x2x2 Strassen blocks, each rank 7, with sharing)
    # Actually, let's use a cleaner approach.
    
    # For 1x4x4: C_j = sum_i A_i B_{i,j}, rank 4
    # For 2x4x4: decompose into four 2x2x2 blocks, each with Strassen rank 7
    # But the blocks share structure. Let's use rank 24 for 2x4x4 (naive: 2*4*4=32, but can do better)
    # Actually, 2x4x4 can be done in rank 24 using a careful block decomposition.
    
    # Let's use a simpler approach:
    # 1x4x4: rank 4
    # 2x4x4: decompose as 2*(2x4x4 blocks for each row) -> but that's not right.
    # 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # Split columns of A and B into two halves of size 2.
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # This is a sum of two 2x2x4 multiplications.
    # Each 2x2x4 has rank... 2*2*4 = 16 naive, or we can use Strassen-like.
    # Actually, 2x2x4 = 2x2x2 (rank 7) + 2x2x2 (rank 7) = rank 14? Not quite, the B blocks are different sizes.
    # Let's just use naive rank for 2x4x4: 32.
    # Total: 4 + 32 = 36. That's better than 44!
    
    # Wait, 1x4x4 rank 4: 
    # C_j = A_0 B_{0j} + A_1 B_{1j} + A_2 B_{2j} + A_3 B_{3j}
    # This is a sum of 4 products, each involving one A element and one B element contributing to one C element.
    # But each product A_i * B_{ij} contributes to C_j. So for fixed i, A_i * B_{ij} for all j.
    # We can write: for each i in 0..3, define M_i = A_i. Then C = sum_i M_i * row_i(B).
    # But in tensor decomposition, each rank-1 term is u * v * w where u is from A, v from B, w from C.
    # For 1x4x4: T_{i, (r,c), j} = delta_{i,r} delta_{c,j}
    # This tensor has rank 4. Decomposition:
    # For k in 0..3:
    #   u_k[i] = delta_{i,k}
    #   v_k[r*4+c] = delta_{r,k} 
    #   w_k[j] = delta_{j,c}? No, c is part of v's index.
    # Hmm, this is tricky. Let me just use the naive rank for 1x4x4 which is 4*4=16? No.
    # Actually, the tensor T_{i, (r,c), j} = delta_{i,r} delta_{c,j} can be written as:
    # sum_{k=0}^3 u_k v_k w_k^T where:
    #   u_k[i] = delta_{i,k}
    #   v_k[r*4+c] = delta_{r,k}
    #   w_k[j] = 1 for all j? No.
    # Let's verify: sum_k delta_{i,k} delta_{r,k} * 1 = delta_{i,r}. But we need delta_{i,r} delta_{c,j}.
    # So w_k must depend on c. But c is part of v's index, not a separate index for w.
    # The issue is that B's index is (r,c) flattened to r*p+c, and we need c=j.
    # So for the term to contribute to C_j, we need the B element to be at position (i,j), i.e., index i*p+j.
    # So for 1x4x4, the nonzero entries of T are at (i, i*4+j, j) for i,j in 0..3.
    # That's 16 nonzero entries. Each rank-1 term covers exactly one such entry (since u, v, w are vectors).
    # So the rank is 16.
    
    # Similarly for 2x4x4: nonzero entries at (i, k*4+j, (i,j)) for i in 0..1, k in 0..3, j in 0..3.
    # That's 2*4*4 = 32 nonzero entries, so rank 32.
    # Total: 16 + 32 = 48. Worse than 44.
    
    # So the block-diagonal approach doesn't help. Let me stick with the 4x(3x2x2) approach with rank 11 each.
    # 3x2x2: A is 3x2, B is 2x2, C is 3x2.
    # Naive rank: 3*2*2 = 12. Can we do better?
    # Using Strassen-like ideas: decompose into 2x2x2 (rank 7) + 1x2x2 (rank 4) = 11.
    # Yes! 3x2x2 = 2x2x2 (rows 0,1 of A) + 1x2x2 (row 2 of A).
    # 2x2x2 has rank 7 (Strassen), 1x2x2 has rank 4 (naive, since 1*2*2=4 and no sharing possible).
    # Total: 7 + 4 = 11.
    
    # So the current approach of 4 blocks of 3x2x2, each rank 11, total 44, is already optimal within this block structure.
    
    # Let me implement the 3x2x2 decomposition with rank 11.
    # For 2x2x2 Strassen (rank 7):
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # i = a*e + b*g -> M1 = (a+b)(e-g) + a*e + b*g... standard Strassen:
    # M1 = (a+b)(e+f)
    # M2 = (a+b)(g+h)
    # M3 = (c+d)*e
    # M4 = (c+d)*h
    # M5 = a*f - b*h
    # M6 = c*g - a*e
    # M7 = b*f - d*h
    # i = M1 - M4 + M6 - M7
    # j = M5 + M7
    # k = M3 + M5
    # l = M1 - M3 + M2 - M6
    
    # For 1x2x2 (rank 4):
    # A = [x, y], B = [[p, q], [r, s]], C = [u, v]
    # u = x*p + y*r
    # v = x*q + y*s
    # Four terms:
    # T1: x * p -> u
    # T2: y * r -> u
    # T3: x * q -> v
    # T4: y * s -> v
    
    # Total 7 + 4 = 11 terms for 3x2x2.
    
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define 3x2x2 factor matrices (rank 11)
    # A: 3x2 -> indices 0..5 (row*2+col)
    # B: 2x2 -> indices 0..3 (row*2+col)
    # C: 3x2 -> indices 0..5 (col*2+row) per trace convention W[k*n+i, r] where k is col, i is row
    # Wait, the trace convention is W[k*n+i, r] where k is column index, i is row index of C.
    # C is 3x2, so rows 0..2, cols 0..1. W index = col*3 + row.
    
    U_322 = np.zeros((6, 11))
    V_322 = np.zeros((4, 11))
    W_322 = np.zeros((6, 11))
    
    # Strassen part (first 7 terms) for rows 0,1 of A
    # A indices: row0col0=0, row0col1=1, row1col0=2, row1col1=3
    # B indices: row0col0=0, row0col1=1, row1col0=2, row1col1=3
    # C indices (col*3+row): col0row0=0, col1row0=3, col0row1=1, col1row1=4, col0row2=2, col1row2=5
    
    # M1 = (a+b)(e+f): U: a(0)+b(1), V: e(0)+f(1), W: contributes to i(0) and l(4)
    U_322[0, 0] = 1; U_322[1, 0] = 1
    V_322[0, 0] = 1; V_322[1, 0] = 1
    W_322[0, 0] = 1; W_322[4, 0] = 1
    
    # M2 = (a+b)(g+h): U: a(0)+b(1), V: g(2)+h(3), W: contributes to l(4)
    U_322[0, 1] = 1; U_322[1, 1] = 1
    V_322[2, 1] = 1; V_322[3, 1] = 1
    W_322[4, 1] = 1
    
    # M3 = (c+d)e: U: c(2)+d(3), V: e(0), W: contributes to k(1)
    U_322[2, 2] = 1; U_322[3, 2] = 1
    V_322[0, 2] = 1
    W_322[1, 2] = 1
    
    # M4 = (c+d)h: U: c(2)+d(3), V: h(3), W: contributes to i(0)
    U_322[2, 3] = 1; U_322[3, 3] = 1
    V_322[3, 3] = 1
    W_322[0, 3] = -1
    
    # M5 = af - bh: U: a(0)f(1) - b(1)h(3), V: f(1) and h(3)... 
    # Actually, M5 = a*f - b*h. This is two products combined.
    # U: a(0) for f part, b(1) for h part
    # V: f(1) for a*f, h(3) for b*h
    # W: contributes to j(3) and k(1)
    # i = M1 - M4 + M6 - M7
    # j = M5 + M7
    # k = M3 + M5
    # l = M1 - M3 + M2 - M6
    
    # M5: a*f contributes to j and k, b*h contributes to j and k with negative sign
    # So for M5 term: U[0]=1, U[1]=-1; V[1]=1, V[3]=1; W[3]=1, W[1]=1
    U_322[0, 4] = 1; U_322[1, 4] = -1
    V_322[1, 4] = 1; V_322[3, 4] = 1
    W_322[3, 4] = 1; W_322[1, 4] = 1
    
    # M6 = cg - ae: c(2)g(2) - a(0)e(0)
    # U: c(2)=1, a(0)=-1; V: g(2)=1, e(0)=1; W: contributes to i(0) with -1 and l(4) with -1
    U_322[2, 5] = 1; U_322[0, 5] = -1
    V_322[2, 5] = 1; V_322[0, 5] = 1
    W_322[0, 5] = -1; W_322[4, 5] = -1
    
    # M7 = bf - dh: b(1)f(1) - d(3)h(3)
    # U: b(1)=1, d(3)=-1; V: f(1)=1, h(3)=1; W: contributes to i(0) with -1 and j(3) with 1
    U_322[1, 6] = 1; U_322[3, 6] = -1
    V_322[1, 6] = 1; V_322[3, 6] = 1
    W_322[0, 6] = -1; W_322[3, 6] = 1
    
    # 1x2x2 part (terms 7-10) for row 2 of A
    # A indices: row2col0=4, row2col1=5
    # B same as before
    # C: col0row2=2, col1row2=5
    
    # T1: x(4) * p(0) -> u(2)
    U_322[4, 7] = 1
    V_322[0, 7] = 1
    W_322[2, 7] = 1
    
    # T2: y(5) * r(2) -> u(2)
    U_322[5, 8] = 1
    V_322[2, 8] = 1
    W_322[2, 8] = 1
    
    # T3: x(4) * q(1) -> v(5)
    U_322[4, 9] = 1
    V_322[1, 9] = 1
    W_322[5, 9] = 1
    
    # T4: y(5) * s(3) -> v(5)
    U_322[5, 10] = 1
    V_322[3, 10] = 1
    W_322[5, 10] = 1
    
    # Now place 4 blocks of 3x2x2 into the 3x4x4 problem
    # Block (br, bc) where br is block row (0..1), bc is block col (0..1)
    # A block: rows br*2..br*2+1, cols bc*2..bc*2+1 -> 2x2, plus row 2
    # Actually, A is 3x4. Split into 4 blocks of 3x2? No.
    # A is 3x4. We can split columns into two halves of size 2.
    # B is 4x4. Split into four 2x2 blocks.
    # C is 3x4. Split columns into two halves of size 2.
    
    # For block (b_row, b_col) where b_row in {0,1}, b_col in {0,1}:
    # A part: all 3 rows, cols b_col*2 to b_col*2+1
    # B part: rows b_row*2 to b_row*2+1, all 4 cols? No.
    # Actually, C[:, b_col*2:b_col*2+2] = sum_{r} A[:, r*2:r*2+2] @ B[r*2:r*2+2, b_col*2:b_col*2+2]
    # So for each (b_row, b_col), we have a 3x2x2 multiplication:
    # A_sub = A[:, b_col*2:b_col*2+2] (3x2)
    # B_sub = B[b_row*2:b_row*2+2, :] ... no, B_sub should be 2x2.
    # Wait, C is 3x4. C[:, j] = sum_i A[:, i] B[i, j]
    # If we group columns of C in pairs, C[:, 2bc:2bc+2] = sum_{i} A[:, i] B[i, 2bc:2bc+2]
    # = sum_{br} sum_{i in br block} A[:, i] B[i, 2bc:2bc+2]
    # So for fixed bc, C[:, 2bc:2bc+2] = A[:, 0:2] @ B[0:2, 2bc:2bc+2] + A[:, 2:4] @ B[2:4, 2bc:2bc+2]
    # Each term is 3x2 @ 2x2 = 3x2.
    # So for each bc (0,1) and each br (0,1), we have a 3x2x2 multiplication.
    # Total 4 blocks, each 3x2x2.
    
    for bc in range(2):
        for br in range(0, 2):
            block_idx = bc * 2 + br
            col_off = block_idx * 11
            
            # A: all 3 rows, cols bc*2 to bc*2+1
            for a_row in range(3):
                for a_col in range(2):
                    local_a = a_row * 2 + a_col
                    global_a = a_row * m + (bc * 2 + a_col)
                    U[global_a, col_off:col_off+11] += U_322[local_a, :]
            
            # B: rows br*2 to br*2+1, cols bc*2 to bc*2+1
            for b_row in range(2):
                for b_col in range(2):
                    local_b = b_row * 2 + b_col
                    global_b = (br * 2 + b_row) * p + (bc * 2 + b_col)
                    V[global_b, col_off:col_off+11] += V_322[local_b, :]
            
            # C: all 3 rows, cols bc*2 to bc*2+1
            # W index = col * n + row
            for c_row in range(3):
                for c_col in range(2):
                    local_c = c_col * 3 + c_row
                    global_c = (bc * 2 + c_col) * n + c_row
                    W[global_c, col_off:col_off+11] += W_322[local_c, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
