import numpy as np
import time
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Build the set of non-zero tensor entries
    # T[a, b, c] = 1 if c = k*n+i where a=i*m+j, b=j*p+k
    # We want to find U, V, W such that sum_d U[a,d]V[b,d]W[c,d] = T[a,b,c]
    
    # The tensor has 40 non-zero entries.
    # We can view this as a system of linear equations for the entries of U, V, W?
    # No, it's bilinear in (U, V) for fixed W, etc.
    # But if we fix the support (which entries are non-zero), it becomes linear?
    # Let's assume a rank R decomposition.
    # We can try to find a solution with integer/half-integer entries.
    
    # Strategy from inspirations:
    # 1. Construct a known valid decomposition (e.g., Strassen-based or Schoolbook).
    # 2. Try to reduce the rank by finding linear dependencies or merging terms.
    
    # Let's start with the schoolbook decomposition (Rank 40)
    # U, V, W are 0/1 matrices with exactly one 1 per row/col in specific patterns.
    
    # Actually, let's try to construct a rank 30 solution as hinted in inspiration 2.
    # Or better, let's use the Strassen construction from inspiration 1 but verify it.
    
    # Let's implement a generic verifier that takes U, V, W and checks if they are valid.
    def verify(U, V, W):
        rank = len(U[0])
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = u_idx(i, j)
                    b = v_idx(j, k)
                    c = w_idx(k, i)
                    val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check a sample of zero entries
        # To be rigorous, we should check all zero entries, but that's 8*20*10 - 40 = 1560 entries.
        # It's fast enough.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Determine if (a,b,c) is a non-zero entry
                    # a = i*m+j, b=j'*p+k, c=k'*n+i'
                    i = a // m
                    j = a % m
                    j_prime = b // p
                    k = b % p
                    k_prime = c // n
                    i_prime = c % n
                    
                    if j == j_prime and k == k_prime and i == i_prime:
                        continue # Non-zero entry, already checked
                    else:
                        val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                        if abs(val) > 1e-9:
                            return False
        return True

    # Let's try to build a rank 30 decomposition.
    # Inspiration 2 suggests:
    # - 4 blocks of 2x2x2 Strassen (7 each) = 28
    # - 2 blocks of 2x2x1 schoolbook (4 each) = 8
    # Total = 36. This matches Inspiration 1.
    
    # Can we do better?
    # Theoretical lower bound for 2x4x5 is not trivial.
    # Let's stick to the rank 36 construction from Inspiration 1, but fix the bugs.
    # The bug in Inspiration 1 was likely in the index mapping or the logic.
    # Let's carefully re-implement the Strassen blocks.
    
    def build_strassen_rank36():
        rank = 36
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        
        def add_strassen_2x2x2(y_indices, z_indices, w_offsets):
            nonlocal col
            # y_indices: [(i,j), (i,j), (i,j), (i,j)] for Y00, Y01, Y10, Y11
            # z_indices: [(j,k), (j,k), (j,k), (j,k)] for Z00, Z01, Z10, Z11
            # w_offsets: [(k,i), (k,i), (k,i), (k,i)] for X00, X01, X10, X11
            
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            w00, w01, w10, w11 = w_offsets
            
            # M1 = (Y00+Y11)(Z00+Z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w00[0], w00[1])][col] += 1
            W[w_idx(w11[0], w11[1])][col] += 1
            col += 1
            
            # M2 = (Y10+Y11)Z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w10[0], w10[1])][col] += 1
            W[w_idx(w11[0], w11[1])][col] -= 1
            col += 1
            
            # M3 = Y00(Z01-Z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w01[0], w01[1])][col] += 1
            W[w_idx(w11[0], w11[1])][col] += 1
            col += 1
            
            # M4 = Y11(Z10-Z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w00[0], w00[1])][col] += 1
            W[w_idx(w10[0], w10[1])][col] += 1
            col += 1
            
            # M5 = (Y00+Y01)Z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w00[0], w00[1])][col] -= 1
            W[w_idx(w01[0], w01[1])][col] += 1
            col += 1
            
            # M6 = (Y10-Y00)(Z00+Z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w11[0], w11[1])][col] += 1
            col += 1
            
            # M7 = (Y01-Y11)(Z10+Z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w00[0], w00[1])][col] += 1
            col += 1

        # We need to partition the matrices.
        # A is 2x4. Split into A_L (2x2) and A_R (2x2).
        # B is 4x5. Split into B_1 (4x2), B_2 (4x2), B_3 (4x1).
        # C is 2x5. Split into C_1 (2x2), C_2 (2x2), C_3 (2x1).
        
        # Block 1: A_L * B_11 (2x2 * 2x2) -> C_11
        # A_L: rows 0,1; cols 0,1
        # B_11: rows 0,1; cols 0,1
        # C_11: rows 0,1; cols 0,1
        
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (0,1), (1,0), (1,1)]
        add_strassen_2x2x2(y1, z1, w1)
        
        # Block 2: A_L * B_21 (2x2 * 2x2) -> C_21
        # A_L: rows 0,1; cols 0,1
        # B_21: rows 2,3; cols 0,1
        # C_21: rows 0,1; cols 2,3
        
        y2 = [(0,0), (0,1), (1,0), (1,1)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(2,0), (2,1), (3,0), (3,1)]
        add_strassen_2x2x2(y2, z2, w2)
        
        # Block 3: A_R * B_12 (2x2 * 2x2) -> C_12
        # A_R: rows 0,1; cols 2,3
        # B_12: rows 0,1; cols 2,3
        # C_12: rows 0,1; cols 2,3
        
        y3 = [(0,2), (0,3), (1,2), (1,3)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (2,1), (3,0), (3,1)]
        add_strassen_2x2x2(y3, z3, w3)
        
        # Block 4: A_R * B_22 (2x2 * 2x2) -> C_22
        # A_R: rows 0,1; cols 2,3
        # B_22: rows 2,3; cols 2,3
        # C_22: rows 0,1; cols 2,3
        
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (2,1), (3,0), (3,1)]
        add_strassen_2x2x2(y4, z4, w4)
        
        # Wait, the W indices for C_11 and C_21 and C_12 and C_22 need to be distinct or added to?
        # C_11 is cols 0,1. C_21 is cols 2,3. C_12 is cols 2,3. C_22 is cols 2,3.
        # This is wrong.
        # A (2x4) * B (4x5) = C (2x5)
        # A = [A_L A_R] where A_L is 2x2, A_R is 2x2.
        # B = [B_1; B_2; B_3] where B_1 is 2x5, B_2 is 2x5, B_3 is 0x5? No.
        # B is 4x5.
        # B = [B_TL B_TR] where B_TL is 2x2, B_TR is 2x3?
        # Let's stick to the column split of B.
        # B = [B_1 B_2 B_3] where B_1 is 4x2, B_2 is 4x2, B_3 is 4x1.
        # C = [C_1 C_2 C_3] where C_1 is 2x2, C_2 is 2x2, C_3 is 2x1.
        
        # C_1 = A * B_1
        # C_2 = A * B_2
        # C_3 = A * B_3
        
        # We can compute C_1 and C_2 using Strassen if we split A and the top/bottom of B.
        # But B_1 is 4x2. We can split B_1 into B_11 (2x2) and B_21 (2x2).
        # Then C_1 = A_L B_11 + A_R B_21.
        # We can compute A_L B_11 and A_R B_21 separately.
        # A_L B_11 is 2x2 * 2x2. Rank 7.
        # A_R B_21 is 2x2 * 2x2. Rank 7.
        # Total for C_1: 14.
        
        # Similarly for C_2:
        # B_2 is 4x2. Split into B_12 (2x2) and B_22 (2x2).
        # C_2 = A_L B_12 + A_R B_22.
        # A_L B_12 is 2x2 * 2x2. Rank 7.
        # A_R B_22 is 2x2 * 2x2. Rank 7.
        # Total for C_2: 14.
        
        # For C_3:
        # B_3 is 4x1. Split into B_13 (2x1) and B_23 (2x1).
        # C_3 = A_L B_13 + A_R B_23.
        # A_L B_13 is 2x2 * 2x1. Schoolbook rank 4.
        # A_R B_23 is 2x2 * 2x1. Schoolbook rank 4.
        # Total for C_3: 8.
        
        # Total Rank = 14 + 14 + 8 = 36.
        
        # Let's re-implement with this logic.
        
        def add_schoolbook_2x2x1(y_indices, z_indices, w_indices):
            nonlocal col
            # y_indices: [(i,j), (i,j), (i,j), (i,j)] for Y00, Y01, Y10, Y11
            # z_indices: [(j,k), (j,k)] for Z00, Z10 (since k is fixed 0 in local 2x1)
            # w_indices: [(k,i), (k,i)] for X00, X10
            y00, y01, y10, y11 = y_indices
            z00, z10 = z_indices
            w00, w10 = w_indices
            
            # M1: Y00 Z00 -> X00
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w00[0], w00[1])][col] += 1
            col += 1
            
            # M2: Y01 Z00 -> X00
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w00[0], w00[1])][col] += 1
            col += 1
            
            # M3: Y10 Z10 -> X10
            U[u_idx(y10[0], y10[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            W[w_idx(w10[0], w10[1])][col] += 1
            col += 1
            
            # M4: Y11 Z10 -> X10
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            W[w_idx(w10[0], w10[1])][col] += 1
            col += 1

        # Reset col
        col = 0
        
        # C_1
        # A_L B_11
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (0,1), (1,0), (1,1)]
        add_strassen_2x2x2(y1, z1, w1)
        
        # A_R B_21
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (0,1), (1,0), (1,1)]
        add_strassen_2x2x2(y2, z2, w2)
        
        # C_2
        # A_L B_12
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (2,1), (3,0), (3,1)]
        add_strassen_2x2x2(y3, z3, w3)
        
        # A_R B_22
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (2,1), (3,0), (3,1)]
        add_strassen_2x2x2(y4, z4, w4)
        
        # C_3
        # A_L B_13
        y5 = [(0,0), (0,1), (1,0), (1,1)]
        z5 = [(0,4), (1,4)]
        w5 = [(4,0), (4,1)]
        add_schoolbook_2x2x1(y5, z5, w5)
        
        # A_R B_23
        y6 = [(0,2), (0,3), (1,2), (1,3)]
        z6 = [(2,4), (3,4)]
        w6 = [(4,0), (4,1)]
        add_schoolbook_2x2x1(y6, z6, w6)
        
        return {"U": U, "V": W, "W": W} # Bug: V is returned as W. Fixed below.
        
    # Let's just build it directly to avoid function scope issues and bugs.
    
    rank = 36
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_2x2x2(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_offsets
        
        # M1
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M2
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] -= 1
        col += 1
        
        # M3
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M4
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        col += 1
        
        # M5
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        col += 1
        
        # M6
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M7
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1
    def add_schoolbook_2x2x1(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z10 = z_indices
        w00, w10 = w_indices
        
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        col += 1
        
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        col += 1
        
        U[u_idx(y10[0], y10[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        col += 1
        
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        col += 1

    # C_1: A_L B_11 + A_R B_21
    add_strassen_2x2x2([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)])
    add_strassen_2x2x2([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(0,1),(1,0),(1,1)])
    
    # C_2: A_L B_12 + A_R B_22
    add_strassen_2x2x2([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)])
    add_strassen_2x2x2([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(2,1),(3,0),(3,1)])
    
    # C_3: A_L B_13 + A_R B_23
    add_schoolbook_2x2x1([(0,0),(0,1),(1,0),(1,1)], [(0,4),(1,4)], [(4,0),(4,1)])
    add_schoolbook_2x2x1([(0,2),(0,3),(1,2),(1,3)], [(2,4),(3,4)], [(4,0),(4,1)])
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 40
    rank = 40
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[u_idx(i,j)][col] = 1
                V[v_idx(j,k)][col] = 1
                W[w_idx(k,i)][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
