import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Max possible rank is n*m*p = 90
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= max_rank:
            raise RuntimeError("Rank exceeded")
        for u_idx, coeff in u_dict.items():
            U[u_idx][col] = coeff
        for v_idx, coeff in v_dict.items():
            V[v_idx][col] = coeff
        for w_idx, coeff in w_dict.items():
            W[w_idx][col] = coeff
        col += 1

    def add_strassen_2x2(u_base, v_base, w_base):
        # u_base, v_base, w_base are tuples of 4 indices: (00, 01, 10, 11)
        u00, u01, u10, u11 = u_base
        v00, v01, v10, v11 = v_base
        w00, w01, w10, w11 = w_base
        
        # M1 = (A00+A11)(B00+B11)
        add_term({u00: 1, u11: 1}, {v00: 1, v11: 1}, {w00: 1, w11: 1})
        
        # M2 = (A10+A11)B00
        add_term({u10: 1, u11: 1}, {v00: 1}, {w10: 1})
        
        # M3 = A00(B10-B11)
        add_term({u00: 1}, {v10: 1, v11: -1}, {w01: 1})
        
        # M4 = A11(B01-B00)
        add_term({u11: 1}, {v01: 1, v00: -1}, {w00: 1, w10: 1})
        
        # M5 = (A00+A01)B11
        add_term({u00: 1, u01: 1}, {v11: 1}, {w00: -1, w01: 1})
        
        # M6 = (A10-A00)(B00+B01)
        add_term({u10: 1, u00: -1}, {v00: 1, v01: 1}, {w11: 1})
        
        # M7 = (A01+A11)(B10-B11)
        add_term({u01: 1, u11: 1}, {v10: 1, v11: -1}, {w00: 1, w11: -1})

    def add_schoolbook_2x2x2x1(u_base, v_base, w_base):
        # u: (00, 01, 10, 11), v: (0, 1), w: (0, 1)
        u00, u01, u10, u11 = u_base
        v0, v1 = v_base
        w0, w1 = w_base
        add_term({u00: 1}, {v0: 1}, {w0: 1})
        add_term({u01: 1}, {v1: 1}, {w0: 1})
        add_term({u10: 1}, {v0: 1}, {w1: 1})
        add_term({u11: 1}, {v1: 1}, {w1: 1})

    def add_schoolbook_1x2x2x2(u_base, v_base, w_base):
        # u: (0, 1), v: (00, 01, 10, 11), w: (0, 1)
        u0, u1 = u_base
        v00, v01, v10, v11 = v_base
        w0, w1 = w_base
        add_term({u0: 1}, {v00: 1}, {w0: 1})
        add_term({u1: 1}, {v10: 1}, {w0: 1})
        add_term({u0: 1}, {v01: 1}, {w1: 1})
        add_term({u1: 1}, {v11: 1}, {w1: 1})

    def add_schoolbook_2x1x2x2(u_base, v_base, w_base):
        # u: (0, 1), v: (0, 1), w: (00, 01, 10, 11)
        u0, u1 = u_base
        v0, v1 = v_base
        w00, w01, w10, w11 = w_base
        add_term({u0: 1}, {v0: 1}, {w00: 1})
        add_term({u0: 1}, {v1: 1}, {w01: 1})
        add_term({u1: 1}, {v0: 1}, {w10: 1})
        add_term({u1: 1}, {v1: 1}, {w11: 1})

    def add_schoolbook_1x2x2x1(u_base, v_base, w_base):
        # u: (0, 1), v: (0, 1), w: (0)
        u0, u1 = u_base
        v0, v1 = v_base
        w0 = w_base[0]
        add_term({u0: 1}, {v0: 1}, {w0: 1})
        add_term({u1: 1}, {v1: 1}, {w0: 1})

    def add_schoolbook_2x2x1x1(u_base, v_base, w_base):
        # u: (00, 01, 10, 11), v: (0), w: (0)
        u00, u01, u10, u11 = u_base
        v0 = v_base[0]
        w0 = w_base[0]
        add_term({u00: 1}, {v0: 1}, {w0: 1})
        add_term({u01: 1}, {v0: 1}, {w0: 1})
        add_term({u10: 1}, {v0: 1}, {w0: 1})
        add_term({u11: 1}, {v0: 1}, {w0: 1})

    def add_schoolbook_1x1x2x2(u_base, v_base, w_base):
        # u: (0), v: (0, 1), w: (00, 01, 10, 11)
        u0 = u_base[0]
        v0, v1 = v_base
        w00, w01, w10, w11 = w_base
        add_term({u0: 1}, {v0: 1}, {w00: 1})
        add_term({u0: 1}, {v0: 1}, {w01: 1}) # Wait, v0 is B[0,0], v1 is B[1,0]? No.
        # v is 1x2? No, v is 2x1?
        # Let's stick to the specific small cases.

    # --- Block 1: A1 (3x2) * B1 (2x6) ---
    # A1 indices: i in 0..2, j in 0..1. U idx: i*5 + j.
    # B1 indices: k in 0..1, l in 0..5. V idx: k*6 + l.
    # C indices: i in 0..2, l in 0..5. W idx: l*3 + i.
    
    # Split B1 into B1a (cols 0..2) and B1b (cols 3..5)
    for b_col_start in [0, 3]:
        # B1a cols: b_col_start, b_col_start+1, b_col_start+2
        # Split A1 into A1a (rows 0,1) and A1b (row 2)
        
        # A1a (2x2) * B1a (2x3)
        # Split B1a into B1a1 (cols 0,1) and B1a2 (col 2)
        
        # A1a (2x2) * B1a1 (2x2) -> Strassen
        u_base = (0*5+0, 0*5+1, 1*5+0, 1*5+1) # A00, A01, A10, A11
        v_base = (0*6+b_col_start, 0*6+b_col_start+1, 1*6+b_col_start, 1*6+b_col_start+1) # B00, B01, B10, B11
        w_base = ((b_col_start)*3+0, (b_col_start)*3+1, (b_col_start)*3+0+3, (b_col_start)*3+1+3) # C00, C01, C10, C11
        # W idx: l*3 + i.
        # C00: l=b_col_start, i=0 -> b_col_start*3 + 0
        # C01: l=b_col_start+1, i=0 -> (b_col_start+1)*3 + 0
        # C10: l=b_col_start, i=1 -> b_col_start*3 + 1
        # C11: l=b_col_start+1, i=1 -> (b_col_start+1)*3 + 1
        w_base = (b_col_start*3+0, (b_col_start+1)*3+0, b_col_start*3+1, (b_col_start+1)*3+1)
        add_strassen_2x2(u_base, v_base, w_base)
        
        # A1a (2x2) * B1a2 (2x1) -> Schoolbook 2x2x2x1
        u_base = (0*5+0, 0*5+1, 1*5+0, 1*5+1)
        v_base = (0*6+b_col_start+2, 1*6+b_col_start+2) # B02, B12
        w_base = (b_col_start*3+2, (b_col_start+1)*3+2) # C02, C12
        # Wait, B1a2 is col 2 of B1a, which is global col b_col_start+2.
        # C is 2x1. C0 is row 0, C1 is row 1.
        # W idx: l*3 + i. l is fixed to b_col_start+2. i is 0 or 1.
        # So W indices are (b_col_start+2)*3 + 0 and (b_col_start+2)*3 + 1.
        w_base = ((b_col_start+2)*3+0, (b_col_start+2)*3+1)
        add_schoolbook_2x2x2x1(u_base, v_base, w_base)
        
        # A1b (1x2) * B1a (2x3) -> Schoolbook
        # A1b: row 2, cols 0,1. U idx: 2*5+0, 2*5+1.
        # B1a: rows 0,1; cols 0,1,2.
        # C: row 2; cols 0,1,2. W idx: l*3 + 2.
        
        # A1b * B1a1 (2x2)
        u_base = (2*5+0, 2*5+1)
        v_base = (0*6+b_col_start, 1*6+b_col_start, 0*6+b_col_start+1, 1*6+b_col_start+1)
        w_base = (b_col_start*3+2, (b_col_start+1)*3+2)
        add_schoolbook_1x2x2x2(u_base, v_base, w_base)
        
        # A1b * B1a2 (2x1)
        u_base = (2*5+0, 2*5+1)
        v_base = (0*6+b_col_start+2, 1*6+b_col_start+2)
        w_base = ((b_col_start+2)*3+2,)
        add_schoolbook_1x2x2x1(u_base, v_base, w_base)

    # --- Block 2: A2 (3x3) * B2 (3x6) ---
    # A2 indices: i in 0..2, j in 2..4. U idx: i*5 + j.
    # B2 indices: k in 2..4, l in 0..5. V idx: k*6 + l.
    # C indices: i in 0..2, l in 0..5. W idx: l*3 + i.
    
    for b_col_start in [0, 3]:
        # B2a cols: b_col_start, b_col_start+1, b_col_start+2
        # A2 (3x3) * B2a (3x3)
        
        # Split A2 into 2x2, 2x1, 1x2, 1x1
        # A11: rows 0,1; cols 2,3.
        # A12: rows 0,1; col 4.
        # A21: row 2; cols 2,3.
        # A22: row 2; col 4.
        
        # Split B2a into 2x2, 2x1, 1x2, 1x1
        # B11: rows 2,3; cols b_col_start, b_col_start+1.
        # B12: rows 2,3; col b_col_start+2.
        # B21: row 4; cols b_col_start, b_col_start+1.
        # B22: row 4; col b_col_start+2.
        
        # C11 (2x2): A11*B11 + A12*B21
        # A11*B11 (2x2x2x2) -> Strassen
        u_base = (0*5+2, 0*5+3, 1*5+2, 1*5+3)
        v_base = (2*6+b_col_start, 2*6+b_col_start+1, 3*6+b_col_start, 3*6+b_col_start+1)
        w_base = (b_col_start*3+0, (b_col_start+1)*3+0, b_col_start*3+1, (b_col_start+1)*3+1)
        add_strassen_2x2(u_base, v_base, w_base)
        
        # A12*B21 (2x1x1x2)
        u_base = (0*5+4, 1*5+4) # A12
        v_base = (4*6+b_col_start, 4*6+b_col_start+1) # B21
        w_base = (b_col_start*3+0, (b_col_start+1)*3+0, b_col_start*3+1, (b_col_start+1)*3+1) # C11
        # C11 is 2x2.
        # A12 is 2x1. B21 is 1x2.
        # C00 = A12[0]*B21[0]
        # C01 = A12[0]*B21[1]
        # C10 = A12[1]*B21[0]
        # C11 = A12[1]*B21[1]
        add_term({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1})
        add_term({u_base[0]: 1}, {v_base[1]: 1}, {w_base[1]: 1})
        add_term({u_base[1]: 1}, {v_base[0]: 1}, {w_base[2]: 1})
        add_term({u_base[1]: 1}, {v_base[1]: 1}, {w_base[3]: 1})
        
        # C12 (2x1): A11*B12 + A12*B22
        # A11*B12 (2x2x2x1)
        u_base = (0*5+2, 0*5+3, 1*5+2, 1*5+3)
        v_base = (2*6+b_col_start+2, 3*6+b_col_start+2) # B12
        w_base = ((b_col_start+2)*3+0, (b_col_start+2)*3+1) # C12
        add_schoolbook_2x2x2x1(u_base, v_base, w_base)
        
        # A12*B22 (2x1x1x1)
        u_base = (0*5+4, 1*5+4)
        v_base = (4*6+b_col_start+2,) # B22
        w_base = ((b_col_start+2)*3+0, (b_col_start+2)*3+1)
        add_term({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1})
        add_term({u_base[1]: 1}, {v_base[0]: 1}, {w_base[1]: 1})
        
        # C21 (1x2): A21*B11 + A22*B21
        # A21*B11 (1x2x2x2)
        u_base = (2*5+2, 2*5+3) # A21
        v_base = (2*6+b_col_start, 2*6+b_col_start+1, 3*6+b_col_start, 3*6+b_col_start+1) # B11
        w_base = (b_col_start*3+2, (b_col_start+1)*3+2) # C21
        add_schoolbook_1x2x2x2(u_base, v_base, w_base)
        
        # A22*B21 (1x1x1x2)
        u_base = (2*5+4,) # A22
        v_base = (4*6+b_col_start, 4*6+b_col_start+1) # B21
        w_base = (b_col_start*3+2, (b_col_start+1)*3+2)
        add_term({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1})
        add_term({u_base[0]: 1}, {v_base[1]: 1}, {w_base[1]: 1})
        
        # C22 (1x1): A21*B12 + A22*B22
        # A21*B12 (1x2x2x1)
        u_base = (2*5+2, 2*5+3)
        v_base = (2*6+b_col_start+2, 3*6+b_col_start+2) # B12
        w_base = ((b_col_start+2)*3+2,) # C22
        add_term({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1})
        add_term({u_base[1]: 1}, {v_base[1]: 1}, {w_base[0]: 1})
        
        # A22*B22 (1x1x1x1)
        u_base = (2*5+4,)
        v_base = (4*6+b_col_start+2,)
        w_base = ((b_col_start+2)*3+2,)
        add_term({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1})

    # Trim matrices to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
