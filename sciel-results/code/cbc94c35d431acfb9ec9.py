import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # Strassen 2x2x2: 7 terms
    # M1 = (a00+a11)(b00+b11) -> c00+c11
    # M2 = (a10+a11)b00 -> c10-c11  
    # M3 = a00(b01-b11) -> c01+c11
    # M4 = a11(b10-b00) -> c00+c10
    # M5 = (a00+a01)b11 -> -c00+c01
    # M6 = (a10-a00)(b00+b01) -> c11
    # M7 = (a01-a11)(b10+b11) -> c00
    
    def strassen_2x2(a_base, b_base, c_base, a_stride, b_stride, c_stride):
        """
        a_base: index of a[0][0]
        b_base: index of b[0][0]  
        c_base: index of c[0][0]
        a_stride: stride between rows in a (for U vector)
        b_stride: stride between rows in b (for V vector)
        c_stride: stride between rows in c (for W vector)
        """
        # M1
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base] = 1
        u[a_base + a_stride + 1] = 1
        v[b_base] = 1
        v[b_base + b_stride + 1] = 1
        w[c_base] = 1
        w[c_base + c_stride + 1] = 1
        add_term(u, v, w)
        
        # M2
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base + a_stride] = 1
        u[a_base + a_stride + 1] = 1
        v[b_base] = 1
        w[c_base + c_stride] = 1
        w[c_base + c_stride + 1] = -1
        add_term(u, v, w)
        
        # M3
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base] = 1
        v[b_base + 1] = 1
        v[b_base + b_stride + 1] = -1
        w[c_base + 1] = 1
        w[c_base + c_stride + 1] = 1
        add_term(u, v, w)
        
        # M4
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base + a_stride + 1] = 1
        v[b_base + b_stride] = 1
        v[b_base] = -1
        w[c_base] = 1
        w[c_base + c_stride] = 1
        add_term(u, v, w)
        
        # M5
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base] = 1
        u[a_base + 1] = 1
        v[b_base + b_stride + 1] = 1
        w[c_base + 1] = 1
        w[c_base] = -1
        add_term(u, v, w)
        
        # M6
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base + a_stride] = 1
        u[a_base] = -1
        v[b_base] = 1
        v[b_base + 1] = 1
        w[c_base + c_stride + 1] = 1
        add_term(u, v, w)
        
        # M7
        u = zero_u(); v = zero_v(); w = zero_w()
        u[a_base + 1] = 1
        u[a_base + a_stride + 1] = -1
        v[b_base + b_stride] = 1
        v[b_base + b_stride + 1] = 1
        w[c_base] = 1
        add_term(u, v, w)

    # Part 1: 4x4 * 4x4 (first 4 columns)
    # Use nested Strassen: outer 7 terms, each inner 7 terms = 49 total
    # A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    
    # For outer Strassen on 4x4:
    # A blocks: A00(0,0:2,2), A01(0,2:2,4), A10(2,0:4,2), A11(2,2:4,4)
    # B blocks: B00(0,0:2,2), B01(0,2:2,4), B10(2,0:4,2), B11(2,2:4,4)
    # C blocks: C00(0,0:2,2), C01(0,2:2,4), C10(2,0:4,2), C11(2,2:4,4)
    
    # Outer Strassen terms for 4x4*4x4:
    # M1 = (A00+A11)(B00+B11) -> C00+C11
    # M2 = (A10+A11)B00 -> C10-C11
    # M3 = A00(B01-B11) -> C01+C11
    # M4 = A11(B10-B00) -> C00+C10
    # M5 = (A00+A01)B11 -> -C00+C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    # For each outer term, we need to compute the product of two 2x2 matrices
    # Each such product uses 7 scalar multiplications
    
    # Let's define helper to get indices
    def a_idx(r, c): return r * m + c  # A is 4x4, m=4
    def b_idx(r, c): return r * p + c  # B is 4x7, p=7
    def c_idx(r, c): return c * n + r  # C is 4x7, n=4, W[k*n+i]
    
    # Outer term 1: (A00+A11)(B00+B11) -> C00+C11
    # A00: rows 0-1, cols 0-1. A11: rows 2-3, cols 2-3
    # B00: rows 0-1, cols 0-1. B11: rows 2-3, cols 2-3
    # C00: rows 0-1, cols 0-1. C11: rows 2-3, cols 2-3
    
    # This is a product of two 2x2 matrices where:
    # Left matrix = A00 + A11 (element-wise sum)
    # Right matrix = B00 + B11 (element-wise sum)
    # Result contributes to C00 + C11
    
    # We can use Strassen on the "virtual" 2x2 matrices
    # The left 2x2 has entries: [A00[0][0]+A11[0][0], A00[0][1]+A11[0][1], A00[1][0]+A11[1][0], A00[1][1]+A11[1][1]]
    # In U vector: a_idx(0,0)+1, a_idx(0,1)+1, a_idx(1,0)+1, a_idx(1,1)+1 for A00 part
    #               a_idx(2,2)+1, a_idx(2,3)+1, a_idx(3,2)+1, a_idx(3,3)+1 for A11 part
    
    # This is getting complex. Let me use a different approach.
    
    # Actually, the cleanest way is to just enumerate all 49 terms explicitly.
    # Each term corresponds to one scalar multiplication in the nested Strassen.
    
    # Let me define the 7 inner Strassen terms as coefficient vectors for a 2x2*2x2 product
    # Inner term i: u_coeff[4], v_coeff[4], w_coeff[4] where indices are 0,1,2,3 for 2x2
    
    inner_u = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [1, 0, -1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    inner_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, -1, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    inner_w = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]
    
    # Outer Strassen for 4x4*4x4:
    # Each outer term specifies which 2x2 blocks to sum and their coefficients
    # outer_a[7]: list of (block, coeff) for A
    # outer_b[7]: list of (block, coeff) for B  
    # outer_c[7]: list of (block, coeff) for C
    # Block 0: top-left (rows 0-1, cols 0-1)
    # Block 1: top-right (rows 0-1, cols 2-3)
    # Block 2: bottom-left (rows 2-3, cols 0-1)
    # Block 3: bottom-right (rows 2-3, cols 2-3)
    
    outer_a = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1), (3, 1)],  # M2
        [(0, 1)],          # M3
        [(3, 1)],          # M4
        [(0, 1), (1, 1)],  # M5
        [(2, 1), (0, -1)], # M6
        [(1, 1), (3, -1)], # M7
    ]
    outer_b = [
        [(0, 1), (3, 1)],  # M1
        [(0, 1)],          # M2
        [(1, 1), (3, -1)], # M3
        [(2, 1), (0, -1)], # M4
        [(3, 1)],          # M5
        [(0, 1), (1, 1)],  # M6
        [(2, 1), (3, 1)],  # M7
    ]
    outer_c = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1), (3, -1)], # M2
        [(1, 1), (3, 1)],  # M3
        [(0, 1), (2, 1)],  # M4
        [(0, -1), (1, 1)], # M5
        [(3, 1)],          # M6
        [(0, 1)],          # M7
    ]
    
    # For each outer term and inner term, create one scalar multiplication term
    for oi in range(7):
        for ii in range(7):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            # U vector: contributions from A blocks
            for (block, coeff) in outer_a[oi]:
                br, bc = block // 2, block % 2
                for lr in range(2):
                    for lc in range(2):
                        inner_idx = lr * 2 + lc
                        r = br * 2 + lr
                        c = bc * 2 + lc
                        u[a_idx(r, c)] += coeff * inner_u[ii][inner_idx]
            
            # V vector: contributions from B blocks
            for (block, coeff) in outer_b[oi]:
                br, bc = block // 2, block % 2
                for lr in range(2):
                    for lc in range(2):
                        inner_idx = lr * 2 + lc
                        r = br * 2 + lr
                        c = bc * 2 + lc
                        v[b_idx(r, c)] += coeff * inner_v[ii][inner_idx]
            
            # W vector: contributions to C blocks
            for (block, coeff) in outer_c[oi]:
                br, bc = block // 2, block % 2
                for lr in range(2):
                    for lc in range(2):
                        inner_idx = lr * 2 + lc
                        r = br * 2 + lr
                        c = bc * 2 + lc
                        w[c_idx(r, c)] += coeff * inner_w[ii][inner_idx]
            
            add_term(u, v, w)
    
    # Part 2: 4x4 * 4x2 (columns 4-5)
    # 4 blocks of 2x2 in A, each multiplied by corresponding 2x2 in B (cols 4-5)
    # Each 2x2*2x2 uses 7 terms
    for i_block in range(2):
        for j_block in range(2):
            # A block: rows i_block*2 to i_block*2+1, cols j_block*2 to j_block*2+1
            # B block: rows j_block*2 to j_block*2+1, cols 4 to 5
            # C block: rows i_block*2 to i_block*2+1, cols 4 to 5
            
            a_base_r = i_block * 2
            a_base_c = j_block * 2
            b_base_r = j_block * 2
            b_base_c = 4
            c_base_r = i_block * 2
            c_base_c = 4
            
            # Use Strassen 2x2x2
            for ii in range(7):
                u = zero_u()
                v = zero_v()
                w = zero_w()
                
                for lr in range(2):
                    for lc in range(2):
                        inner_idx = lr * 2 + lc
                        r = a_base_r + lr
                        c = a_base_c + lc
                        u[a_idx(r, c)] += inner_u[ii][inner_idx]
                        
                        r = b_base_r + lr
                        c = b_base_c + lc
                        v[b_idx(r, c)] += inner_v[ii][inner_idx]
                        
                        r = c_base_r + lr
                        c = c_base_c + lc
                        w[c_idx(r, c)] += inner_w[ii][inner_idx]
                
                add_term(u, v, w)
    
    # Part 3: 4x4 * 4x1 (column 6) - schoolbook
    for i in range(4):
        for j in range(4):
            u = zero_u()
            v = zero_v()
            w = zero_w()
            u[a_idx(i, j)] = 1
            v[b_idx(j, 6)] = 1
            w[c_idx(i, 6)] = 1
            add_term(u, v, w)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
