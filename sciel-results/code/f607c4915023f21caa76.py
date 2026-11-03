"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # rank = 85 expected
    # We will build U, V, W lists of lists.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        """
        u_idx: index in A (0 to n*m-1)
        v_idx: index in B (0 to m*p-1)
        w_idx: index in C (0 to p*n-1)
        coeff: integer
        """
        # We accumulate columns
        # It's easier to build a list of (u, v, w, coeff) tuples and then convert
        pass

    terms = []
    
    def add_strassen_2x2x2(a_base, b_base, c_base):
        """
        Adds rank 7 terms for 2x2 * 2x2.
        a_base: (i, j) top-left of A block (2x2)
        b_base: (j, k) top-left of B block (2x2)
        c_base: (i, k) top-left of C block (2x2)
        
        A indices:
        a00 = a_base[0]*m + a_base[1]
        a01 = a_base[0]*m + a_base[1] + 1
        a10 = (a_base[0]+1)*m + a_base[1]
        a11 = (a_base[0]+1)*m + a_base[1] + 1
        
        B indices:
        b00 = b_base[0]*p + b_base[1]
        b01 = b_base[0]*p + b_base[1] + 1
        b10 = (b_base[0]+1)*p + b_base[1]
        b11 = (b_base[0]+1)*p + b_base[1] + 1
        
        C indices (W convention: k*n + i):
        c00 = c_base[1]*n + c_base[0]
        c01 = (c_base[1]+1)*n + c_base[0]
        c10 = c_base[1]*n + (c_base[0]+1)
        c11 = (c_base[1]+1)*n + (c_base[0]+1)
        """
        a00, a01, a10, a11 = (
            a_base[0]*m + a_base[1],
            a_base[0]*m + a_base[1] + 1,
            (a_base[0]+1)*m + a_base[1],
            (a_base[0]+1)*m + a_base[1] + 1
        )
        b00, b01, b10, b11 = (
            b_base[0]*p + b_base[1],
            b_base[0]*p + b_base[1] + 1,
            (b_base[0]+1)*p + b_base[1],
            (b_base[0]+1)*p + b_base[1] + 1
        )
        c00, c01, c10, c11 = (
            c_base[1]*n + c_base[0],
            (c_base[1]+1)*n + c_base[0],
            c_base[1]*n + (c_base[0]+1),
            (c_base[1]+1)*n + (c_base[0]+1)
        )
        
        # Strassen 7 terms
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add terms to the global list
        # Each term is (u_indices, v_indices, w_indices, coeffs)
        # Actually, U, V, W are matrices. A rank 1 term is u * v * w.
        # Strassen terms are sums of products.
        # M1 = (a00+a11)*(b00+b11) is not rank 1. It's (a00+a11) * (b00+b11).
        # This corresponds to U having 1 in a00 and a11, V having 1 in b00 and b11.
        # So one column in U, V, W.
        
        # Term 1: U: a00=1, a11=1. V: b00=1, b11=1. W: c00=1, c01=1, c10=1, c11=1 (with signs)
        # Wait, W entries for M1:
        # C00 gets +1. C11 gets +1. C01 gets 0. C10 gets 0.
        # So W: c00=1, c11=1.
        
        # Let's define the 7 columns for Strassen
        # Col 1:
        u1 = {a00: 1, a11: 1}
        v1 = {b00: 1, b11: 1}
        w1 = {c00: 1, c11: 1}
        
        # Col 2:
        u2 = {a10: 1, a11: 1}
        v2 = {b00: 1}
        w2 = {c10: 1, c11: -1}
        
        # Col 3:
        u3 = {a00: 1}
        v3 = {b01: 1, b11: -1}
        w3 = {c01: 1, c11: 1}
        
        # Col 4:
        u4 = {a11: 1}
        v4 = {b10: 1, b00: -1}
        w4 = {c00: -1, c10: 1}
        
        # Col 5:
        u5 = {a00: 1, a01: 1}
        v5 = {b11: 1}
        w5 = {c00: -1, c01: 1}
        
        # Col 6:
        u6 = {a10: 1, a00: -1}
        v6 = {b00: 1, b01: 1}
        w6 = {c11: 1}
        
        # Col 7:
        u7 = {a01: 1, a11: -1}
        v7 = {b10: 1, b11: 1}
        w7 = {c00: 1, c11: -1}
        
        for u, v, w in [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]:
            terms.append((u, v, w))

    def add_schoolbook_2x2x1(a_base, b_base, c_base):
        """
        2x2 * 2x1 -> 2x1
        a_base: (i, j)
        b_base: (j, k)
        c_base: (i, k)
        """
        a00 = a_base[0]*m + a_base[1]
        a01 = a_base[0]*m + a_base[1] + 1
        a10 = (a_base[0]+1)*m + a_base[1]
        a11 = (a_base[0]+1)*m + a_base[1] + 1
        
        b00 = b_base[0]*p + b_base[1]
        b10 = (b_base[0]+1)*p + b_base[1]
        
        c0 = c_base[1]*n + c_base[0]
        c1 = c_base[1]*n + (c_base[0]+1)
        
        # C0 = a00*b00 + a01*b10
        # C1 = a10*b00 + a11*b10
        
        # 4 terms
        terms.append(({a00: 1}, {b00: 1}, {c0: 1}))
        terms.append(({a01: 1}, {b10: 1}, {c0: 1}))
        terms.append(({a10: 1}, {b00: 1}, {c1: 1}))
        terms.append(({a11: 1}, {b10: 1}, {c1: 1}))

    def add_schoolbook_1x2x7(a_base, b_base, c_base):
        """
        1x2 * 2x7 -> 1x7
        a_base: (i, j)
        b_base: (j, k)
        c_base: (i, k)
        """
        a0 = a_base[0]*m + a_base[1]
        a1 = a_base[0]*m + a_base[1] + 1
        
        # B is 2x7
        # Rows j, j+1. Cols k..k+6
        b_row0 = b_base[0]*p + b_base[1]
        b_row1 = (b_base[0]+1)*p + b_base[1]
        
        c_row = c_base[1]*n + c_base[0]
        
        for k in range(7):
            b0 = b_row0 + k
            b1 = b_row1 + k
            c = c_row + k*n # Wait, c index is k*n + i.
            # c_base is (i, k). c_row calculated above is for k=c_base[1].
            # We need c for each k.
            # c_idx = (k)*n + i.
            # i is c_base[0].
            i = c_base[0]
            c_idx = k*n + i
            
            terms.append(({a0: 1}, {b0: 1}, {c_idx: 1}))
            terms.append(({a1: 1}, {b1: 1}, {c_idx: 1}))

    def add_schoolbook_2x1x7(a_base, b_base, c_base):
        """
        2x1 * 1x7 -> 2x7
        a_base: (i, j)
        b_base: (j, k)
        c_base: (i, k)
        """
        a0 = a_base[0]*m + a_base[1]
        a1 = (a_base[0]+1)*m + a_base[1]
        
        b0 = b_base[0]*p + b_base[1]
        
        i0 = c_base[0]
        i1 = c_base[0] + 1
        
        for k in range(7):
            c_idx0 = k*n + i0
            c_idx1 = k*n + i1
            
            terms.append(({a0: 1}, {b0: 1}, {c_idx0: 1}))
            terms.append(({a1: 1}, {b0: 1}, {c_idx1: 1}))

    def add_schoolbook_1x1x7(a_base, b_base, c_base):
        """
        1x1 * 1x7 -> 1x7
        """
        a0 = a_base[0]*m + a_base[1]
        b0 = b_base[0]*p + b_base[1]
        i = c_base[0]
        
        for k in range(7):
            b_idx = b0 + k
            c_idx = k*n + i
            terms.append(({a0: 1}, {b_idx: 1}, {c_idx: 1}))

    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # Split B into [0:2, 0:2], [0:2, 2:4], [0:2, 4:5]
    add_strassen_2x2x2((0,0), (0,0), (0,0))
    add_strassen_2x2x2((0,0), (0,2), (0,2))
    add_schoolbook_2x2x1((0,0), (0,4), (0,4))
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:7]
    # Split B into [2:4, 0:2], [2:4, 2:4], [2:4, 4:5]
    add_strassen_2x2x2((0,2), (2,0), (0,0))
    add_strassen_2x2x2((0,2), (2,2), (0,2))
    add_schoolbook_2x2x1((0,2), (2,4), (0,4))
    
    # Block 3: A[0:2, 4:5] * B[4:5, 0:7]
    # 2x1 * 1x7
    add_schoolbook_2x1x7((0,4), (4,0), (0,0))
    
    # Block 4: A[2:3, 0:2] * B[0:2, 0:7]
    # 1x2 * 2x7
    add_schoolbook_1x2x7((2,0), (0,0), (2,0))
    
    # Block 5: A[2:3, 2:4] * B[2:4, 0:7]
    # 1x2 * 2x7
    add_schoolbook_1x2x7((2,2), (2,0), (2,0))
    
    # Block 6: A[2:3, 4:5] * B[4:5, 0:7]
    # 1x1 * 1x7
    add_schoolbook_1x1x7((2,4), (4,0), (2,0))

    rank = len(terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
