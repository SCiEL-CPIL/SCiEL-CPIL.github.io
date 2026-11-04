"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid matrix multiplication tensor decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Condition: for all i,k,j,l: sum_c W[k*n+i,c]*U[i*m+j,c]*V[l*p+k,c] = delta(j,l)
        """
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        s = 0.0
                        for c in range(rank):
                            s += W[k * n + i][c] * U[i * m + j][c] * V[l * p + k][c]
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def build_strassen_444():
        """Build the 49 rank-1 terms for 4x4x4 multiplication using Strassen recursion."""
        terms = []
        
        # Define the 2x2x2 Strassen basis terms
        # Each term is a dict with 'A', 'B', 'C' keys, mapping local (i,j) to coeff
        # Local indices are 0,1 for 2x2 blocks
        
        # We need to map the 4x4x4 Strassen steps to 2x2x2 Strassen steps.
        # The 4x4x4 Strassen steps are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each M is a product of two 2x2 matrices.
        # We can decompose each M using the 2x2x2 Strassen basis.
        
        # 2x2x2 Strassen basis terms (u, v, w)
        # u, v, w are dicts mapping (i,j) to coeff
        s_terms = []
        
        # T1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        s_terms.append((u1, v1, w1))
        
        # T2: (a10+a11)b00 -> c10+=1, c11-=1
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1, (1,1): -1}
        s_terms.append((u2, v2, w2))
        
        # T3: a00(b01-b11) -> c01+=1, c11+=1
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}
        s_terms.append((u3, v3, w3))
        
        # T4: a11(b10-b00) -> c00+=1, c10+=1
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        s_terms.append((u4, v4, w4))
        
        # T5: (a00+a01)b11 -> c00-=1, c01+=1
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}
        s_terms.append((u5, v5, w5))
        
        # T6: (a10-a00)(b00+b01) -> c11+=1
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        s_terms.append((u6, v6, w6))
        
        # T7: (a01-a11)(b10+b11) -> c00+=1
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        s_terms.append((u7, v7, w7))

        # Now, for each of the 7 M terms in 4x4x4 Strassen, we add 7 terms from s_terms.
        # We need to map the local 2x2 indices to the global 4x4 indices.
        
        # M1: (A00+A11)(B00+B11)
        # A part: A00 (global 0:2, 0:2) and A11 (global 2:4, 2:4)
        # B part: B00 (global 0:2, 0:2) and B11 (global 2:4, 2:4)
        # This is a sum of two 2x2x2 multiplications? No, it's one product of sums.
        # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # We can treat (A00+A11) as a virtual A matrix and (B00+B11) as virtual B matrix.
        # The virtual A has entries:
        #   vA[0,0] = A[0,0] + A[2,2]
        #   vA[0,1] = A[0,1] + A[2,3]
        #   vA[1,0] = A[1,0] + A[3,2]
        #   vA[1,1] = A[1,1] + A[3,3]
        # Similarly for vB.
        # And the result vC contributes to C00 and C11?
        # In 4x4x4 Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes to C00 (block 0:2, 0:2) and C11 (block 2:4, 2:4).
        
        # To add the terms, we iterate over the 7 s_terms.
        # For each s_term (u, v, w), we map the local indices to global indices for the specific M term.
        
        def add_m_term(m_idx, a_blocks, b_blocks, c_blocks, signs):
            # a_blocks: list of (global_row_offset, global_col_offset, sign)
            # b_blocks: list of (global_row_offset, global_col_offset, sign)
            # c_blocks: list of (global_row_offset, global_col_offset, sign)
            # This is getting complicated. Let's use a direct mapping.
            
            pass

        # Let's do it explicitly for each M.
        
        # Helper to add a term to the list
        def add_term(u_dict, v_dict, w_dict):
            # u_dict: {(global_i, global_j): coeff}
            # v_dict: {(global_j, global_k): coeff}
            # w_dict: {(global_k, global_i): coeff}
            terms.append((u_dict, v_dict, w_dict))

        # M1: (A00+A11)(B00+B11)
        # A coeffs: A00 (1), A11 (1)
        # B coeffs: B00 (1), B11 (1)
        # C coeffs: C00 (1), C11 (1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # Map A local (i,j) to global
            # A00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # A11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in u.items():
                u_new[(li, lj)] = coeff # From A00
                u_new[(li+2, lj+2)] = coeff # From A11
                
            # Map B local (i,j) to global
            # B00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # B11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in v.items():
                v_new[(li, lj)] = coeff # From B00
                v_new[(li+2, lj+2)] = coeff # From B11
                
            # Map C local (i,j) to global
            # C00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # C11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in w.items():
                w_new[(li, lj)] = coeff # From C00
                w_new[(li+2, lj+2)] = coeff # From C11
                
            add_term(u_new, v_new, w_new)

        # M2: (A10+A11)B00
        # A: A10 (1), A11 (1)
        # B: B00 (1)
        # C: C10 (1), C11 (-1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # A10: global (2:4, 0:2). Local (i,j) -> Global (i+2, j)
            # A11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in u.items():
                u_new[(li+2, lj)] = coeff # From A10
                u_new[(li+2, lj+2)] = coeff # From A11
                
            # B00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            for (li, lj), coeff in v.items():
                v_new[(li, lj)] = coeff # From B00
                
            # C10: global (2:4, 0:2). Local (i,j) -> Global (i+2, j)
            # C11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in w.items():
                w_new[(li+2, lj)] = coeff # From C10
                w_new[(li+2, lj+2)] = -coeff # From C11 (sign -1)
                
            add_term(u_new, v_new, w_new)

        # M3: A00(B01-B11)
        # A: A00 (1)
        # B: B01 (1), B11 (-1)
        # C: C01 (1), C11 (1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # A00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            for (li, lj), coeff in u.items():
                u_new[(li, lj)] = coeff # From A00
                
            # B01: global (0:2, 2:4). Local (i,j) -> Global (i, j+2)
            # B11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in v.items():
                v_new[(li, lj+2)] = coeff # From B01
                v_new[(li+2, lj+2)] = -coeff # From B11
                
            # C01: global (0:2, 2:4). Local (i,j) -> Global (i, j+2)
            # C11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in w.items():
                w_new[(li, lj+2)] = coeff # From C01
                w_new[(li+2, lj+2)] = coeff # From C11
                
            add_term(u_new, v_new, w_new)

        # M4: A11(B10-B00)
        # A: A11 (1)
        # B: B10 (1), B00 (-1)
        # C: C00 (1), C10 (1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # A11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in u.items():
                u_new[(li+2, lj+2)] = coeff # From A11
                
            # B10: global (2:4, 0:2). Local (i,j) -> Global (i+2, j)
            # B00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            for (li, lj), coeff in v.items():
                v_new[(li+2, lj)] = coeff # From B10
                v_new[(li, lj)] = -coeff # From B00
                
            # C00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # C10: global (2:4, 0:2). Local (i,j) -> Global (i+2, j)
            for (li, lj), coeff in w.items():
                w_new[(li, lj)] = coeff # From C00
                w_new[(li+2, lj)] = coeff # From C10
                
            add_term(u_new, v_new, w_new)

        # M5: (A00+A01)B11
        # A: A00 (1), A01 (1)
        # B: B11 (1)
        # C: C00 (-1), C01 (1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # A00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # A01: global (0:2, 2:4). Local (i,j) -> Global (i, j+2)
            for (li, lj), coeff in u.items():
                u_new[(li, lj)] = coeff # From A00
                u_new[(li, lj+2)] = coeff # From A01
                
            # B11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in v.items():
                v_new[(li+2, lj+2)] = coeff # From B11
                
            # C00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # C01: global (0:2, 2:4). Local (i,j) -> Global (i, j+2)
            for (li, lj), coeff in w.items():
                w_new[(li, lj)] = -coeff # From C00
                w_new[(li, lj+2)] = coeff # From C01
                
            add_term(u_new, v_new, w_new)

        # M6: (A10-A00)(B00+B01)
        # A: A10 (1), A00 (-1)
        # B: B00 (1), B01 (1)
        # C: C11 (1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # A10: global (2:4, 0:2). Local (i,j) -> Global (i+2, j)
            # A00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            for (li, lj), coeff in u.items():
                u_new[(li+2, lj)] = coeff # From A10
                u_new[(li, lj)] = -coeff # From A00
                
            # B00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            # B01: global (0:2, 2:4). Local (i,j) -> Global (i, j+2)
            for (li, lj), coeff in v.items():
                v_new[(li, lj)] = coeff # From B00
                v_new[(li, lj+2)] = coeff # From B01
                
            # C11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in w.items():
                w_new[(li+2, lj+2)] = coeff # From C11
                
            add_term(u_new, v_new, w_new)

        # M7: (A01-A11)(B10+B11)
        # A: A01 (1), A11 (-1)
        # B: B10 (1), B11 (1)
        # C: C00 (1)
        
        for u, v, w in s_terms:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # A01: global (0:2, 2:4). Local (i,j) -> Global (i, j+2)
            # A11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in u.items():
                u_new[(li, lj+2)] = coeff # From A01
                u_new[(li+2, lj+2)] = -coeff # From A11
                
            # B10: global (2:4, 0:2). Local (i,j) -> Global (i+2, j)
            # B11: global (2:4, 2:4). Local (i,j) -> Global (i+2, j+2)
            for (li, lj), coeff in v.items():
                v_new[(li+2, lj)] = coeff # From B10
                v_new[(li+2, lj+2)] = coeff # From B11
                
            # C00: global (0:2, 0:2). Local (i,j) -> Global (i, j)
            for (li, lj), coeff in w.items():
                w_new[(li, lj)] = coeff # From C00
                
            add_term(u_new, v_new, w_new)

        return terms

    # Build the 49 terms for 4x4x4
    strassen_terms = build_strassen_444()
    
    # Build the 48 schoolbook terms for 4x4x3 (columns 4,5,6 of B)
    schoolbook_terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                schoolbook_terms.append((u, v, w))
                
    all_terms = strassen_terms + schoolbook_terms
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][c] = coeff
            
    if check_decomposition(U, V, W, tol=1e-9):
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
        
    return best
# EVOLVE-BLOCK-END
