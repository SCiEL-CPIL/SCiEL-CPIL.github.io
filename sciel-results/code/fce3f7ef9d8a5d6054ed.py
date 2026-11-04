import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    strassen_terms = [
        # (U_coeffs [a,b,c,d], V_coeffs [e,f,g,h], W_coeffs [C11, C12, C21, C22])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2 -> C21
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3 -> C12
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4 -> C11, C21
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # M5 -> C11 (neg), C12
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6 -> C22
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7 -> C11
    ]
    # Correction for M5, M6, M7 based on standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify the W coeffs for each M_t
    # M1: C11 +1, C22 +1
    # M2: C21 +1
    # M3: C12 +1
    # M4: C11 +1, C21 +1
    # M5: C11 -1, C12 +1
    # M6: C22 +1
    # M7: C11 +1
    
    # Wait, M5 is (a+b)h.
    # C11 has -M5. C12 has +M5.
    # So for M5 term: W_coeffs [ -1, 1, 0, 0 ]
    # M6 is (c-a)(e+f).
    # C22 has +M6.
    # So for M6 term: W_coeffs [ 0, 0, 0, 1 ]
    # M7 is (b-d)(g+h).
    # C11 has +M7.
    # So for M7 term: W_coeffs [ 1, 0, 0, 0 ]
    
    # Let's update the list
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]

    def add_strassen_term(i, j, k, u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        # i, j, k are block indices
        # A block i,j is 2x2
        # B block j,k is 2x2
        # C block i,k is 2x2
        
        # U indices: A[i, j]
        # A global rows: 2*i, 2*i+1
        # A global cols: 2*j, 2*j+1
        # U row index: a_global * m + b_global
        # a_global in {2i, 2i+1}
        # b_global in {2j, 2j+1}
        
        # V indices: B[j, k]
        # B global rows: 2*j, 2*j+1
        # B global cols: 2*k, 2*k+1
        # V row index: c_global * p + d_global
        # c_global in {2j, 2j+1}
        # d_global in {2k, 2k+1}
        
        # W indices: C[i, k]
        # C global rows: 2*i, 2*i+1
        # C global cols: 2*k, 2*k+1
        # W row index: col_global * n + row_global
        # col_global in {2k, 2k+1}
        # row_global in {2i, 2i+1}
        
        for a in range(2):
            for b in range(2):
                u_val = u_coeffs[a * 2 + b]
                if u_val != 0:
                    u_row = (2 * i + a) * m + (2 * j + b)
                    U[u_row][col] += u_val
                    
        for c in range(2):
            for d in range(2):
                v_val = v_coeffs[c * 2 + d]
                if v_val != 0:
                    v_row = (2 * j + c) * p + (2 * k + d)
                    V[v_row][col] += v_val
                    
        for a in range(2):
            for b in range(2):
                w_val = w_coeffs[a * 2 + b]
                if w_val != 0:
                    w_row = (2 * k + b) * n + (2 * i + a)
                    W[w_row][col] += w_val
                    
        col += 1

    def add_outer_term(i, j, k, u_vec, v_vec):
        # u_vec is length 2 (A[i, j] is 2x1)
        # v_vec is length 2 (B[j, k] is 1x2)
        # Result is 2x2
        nonlocal col
        # A block i,j (2x1)
        # Rows: 2i, 2i+1. Col: 2j (only 1 col)
        # B block j,k (1x2)
        # Row: 2j (only 1 row). Cols: 2k, 2k+1
        
        # U
        for a in range(2):
            u_val = u_vec[a]
            if u_val != 0:
                u_row = (2 * i + a) * m + (2 * j)
                U[u_row][col] += u_val
                
        # V
        for b in range(2):
            v_val = v_vec[b]
            if v_val != 0:
                v_row = (2 * j) * p + (2 * k + b)
                V[v_row][col] += v_val
                
        # W
        # C[i, k] is 2x2
        # C[a, b] = u_vec[a] * v_vec[b]
        # This is a single rank-1 term?
        # No, the tensor is sum over r of U_r V_r W_r.
        # Here we are adding ONE term to the decomposition.
        # The term is U * V * W.
        # U is 2x1, V is 1x2.
        # The product U V is 2x2.
        # W should be the identity?
        # C = U V.
        # So W coefficients are 1 for all C[a, b].
        
        for a in range(2):
            for b in range(2):
                w_row = (2 * k + b) * n + (2 * i + a)
                W[w_row][col] += 1
                
        col += 1

    # Loop over blocks
    for i in range(2):
        for k in range(3):
            # Term 1: A[i,0] B[0,k] (2x2 * 2x2)
            for t in strassen_terms:
                add_strassen_term(i, 0, k, t[0], t[1], t[2])
                
            # Term 2: A[i,1] B[1,k] (2x2 * 2x2)
            for t in strassen_terms:
                add_strassen_term(i, 1, k, t[0], t[1], t[2])
                
            # Term 3: A[i,2] B[2,k] (2x1 * 1x2)
            # A[i,2] is 2x1. B[2,k] is 1x2.
            # We can decompose this into 4 rank-1 terms?
            # No, it's a single outer product.
            # But in the tensor decomposition, we need to represent the multiplication.
            # The multiplication is C[a,b] = A[a,0] B[0,b].
            # This is a single rank-1 term in the tensor?
            # Yes, if we define U = A, V = B, W = I.
            # But U is 2x1, V is 1x2.
            # So we add 1 term.
            
            # Wait, the rank of 2x1x2 is 4?
            # No, the rank of the multiplication tensor M_{2,1,2} is 4.
            # Why?
            # M_{2,1,2} is the sum of 4 rank-1 tensors?
            # C_{ab} = A_a B_b.
            # This is a single rank-1 tensor in the space of matrices?
            # No, the tensor is T_{a, 0, b} = A_a B_b.
            # This is a rank-1 tensor in the 3-way tensor space?
            # Yes.
            # So we only need 1 term.
            
            # However, my add_outer_term adds 1 term.
            # Let's check the rank count.
            # If I use 1 term for 2x1x2, total rank per (i,k) is 7+7+1 = 15.
            # Total rank 6 * 15 = 90.
            # But schoolbook is 120.
            # Is rank 90 possible?
            # Lower bound for 2x1x2 is 4?
            # No, the rank of M_{m,1,p} is mp.
            # For m=2, p=2, rank is 4.
            # So I need 4 terms.
            
            # How to represent 2x1x2 with 4 terms?
            # C_{ab} = A_a B_b.
            # We can write this as sum of 4 terms?
            # Term 1: A_0 B_0 -> C_00
            # Term 2: A_0 B_1 -> C_01
            # Term 3: A_1 B_0 -> C_10
            # Term 4: A_1 B_1 -> C_11
            
            # So we call add_outer_term 4 times?
            # No, add_outer_term adds 1 term that covers the whole 2x2.
            # That would be rank 1.
            # But rank 1 is not enough for 2x1x2.
            
            # So I must decompose the 2x1x2 multiplication into 4 rank-1 terms.
            
            # Term for C_00: U=[1,0], V=[1,0], W=[1,0,0,0]
            # Term for C_01: U=[1,0], V=[0,1], W=[0,1,0,0]
            # Term for C_10: U=[0,1], V=[1,0], W=[0,0,1,0]
            # Term for C_11: U=[0,1], V=[0,1], W=[0,0,0,1]
            
            for a in range(2):
                for b in range(2):
                    u_vec = [0, 0]
                    u_vec[a] = 1
                    v_vec = [0, 0]
                    v_vec[b] = 1
                    add_outer_term(i, 2, k, u_vec, v_vec)

    return {"U": U, "V": V, "W": W}
