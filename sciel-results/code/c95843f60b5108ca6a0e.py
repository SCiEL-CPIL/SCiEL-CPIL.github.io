import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_lf(lf1, lf2, sign=1):
        res = dict(lf1)
        for k, v in lf2.items():
            res[k] = res.get(k, 0) + sign * v
            if res[k] == 0:
                del res[k]
        return res
        
    def sub_lf(lf1, lf2):
        return add_lf(lf1, lf2, -1)

    def block_2x2_lf(br, bc):
        base = br*8 + bc*2
        idxs = [base, base+1, base+4, base+5]
        return [ {idx: 1} for idx in idxs ]
        
    A00_2x2 = block_2x2_lf(0,0)
    A01_2x2 = block_2x2_lf(0,1)
    A10_2x2 = block_2x2_lf(1,0)
    A11_2x2 = block_2x2_lf(1,1)
    
    B00_2x2 = block_2x2_lf(0,0)
    B01_2x2 = block_2x2_lf(0,1)
    B10_2x2 = block_2x2_lf(1,0)
    B11_2x2 = block_2x2_lf(1,1)
    
    def add_2x2_lf(M1, M2, sign=1):
        res = []
        for i in range(4):
            res.append(add_lf(M1[i], M2[i], sign))
        return res
        
    def sub_2x2_lf(M1, M2):
        return add_2x2_lf(M1, M2, -1)

    # C_map: for each Z entry (0..3), list of (c_br, c_bc, c_sign)
    steps_data = [
        {
            "X": add_2x2_lf(A00_2x2, A11_2x2),
            "Y": add_2x2_lf(B00_2x2, B11_2x2),
            "C_map": [
                [(0,0, 1)], # Z00 -> C00
                [(0,1, 1)], # Z01 -> C01
                [(1,0, 1)], # Z10 -> C10
                [(1,1, 1)]  # Z11 -> C11
            ]
        },
        {
            "X": add_2x2_lf(A10_2x2, A11_2x2),
            "Y": B00_2x2,
            "C_map": [
                [(1,0, 1)], # Z00 -> C10
                [],         # Z01 -> None
                [(1,0, 1)], # Z10 -> C10
                [(1,1, -1)] # Z11 -> C11
            ]
        },
        {
            "X": A00_2x2,
            "Y": sub_2x2_lf(B01_2x2, B11_2x2),
            "C_map": [
                [],         # Z00 -> None
                [(0,1, 1)], # Z01 -> C01
                [],         # Z10 -> None
                [(1,1, 1)]  # Z11 -> C11
            ]
        },
        {
            "X": A11_2x2,
            "Y": sub_2x2_lf(B10_2x2, B00_2x2),
            "C_map": [
                [(0,0, 1)], # Z00 -> C00
                [],         # Z01 -> None
                [(1,0, 1)], # Z10 -> C10
                []          # Z11 -> None
            ]
        },
        {
            "X": add_2x2_lf(A00_2x2, A01_2x2),
            "Y": B11_2x2,
            "C_map": [
                [(0,0, -1)], # Z00 -> C00
                [(0,1, 1)],  # Z01 -> C01
                [],          # Z10 -> None
                []           # Z11 -> None
            ]
        },
        {
            "X": sub_2x2_lf(A10_2x2, A00_2x2),
            "Y": add_2x2_lf(B00_2x2, B01_2x2),
            "C_map": [
                [],          # Z00 -> None
                [],          # Z01 -> None
                [],          # Z10 -> None
                [(1,1, 1)]   # Z11 -> C11
            ]
        },
        {
            "X": sub_2x2_lf(A01_2x2, A11_2x2),
            "Y": add_2x2_lf(B10_2x2, B11_2x2),
            "C_map": [
                [(0,0, 1)],  # Z00 -> C00
                [],          # Z01 -> None
                [],          # Z10 -> None
                []           # Z11 -> None
            ]
        }
    ]
    
    z_coeffs = [
        {0: 1, 3: 1}, # S1
        {2: 1, 3: -1}, # S2
        {1: 1, 3: 1}, # S3
        {0: 1, 2: 1}, # S4
        {0: -1, 1: 1}, # S5
        {3: 1}, # S6
        {0: 1} # S7
    ]
    
    col = 0
    for step in steps_data:
        X = step["X"]
        Y = step["Y"]
        
        S1_u = add_lf(X[0], X[3])
        S1_v = add_lf(Y[0], Y[3])
        
        S2_u = add_lf(X[2], X[3])
        S2_v = Y[0]
        
        S3_u = X[0]
        S3_v = sub_lf(Y[1], Y[3])
        
        S4_u = X[3]
        S4_v = sub_lf(Y[2], Y[0])
        
        S5_u = add_lf(X[0], X[1])
        S5_v = Y[3]
        
        S6_u = sub_lf(X[2], X[0])
        S6_v = add_lf(Y[0], Y[1])
        
        S7_u = sub_lf(X[1], X[3])
        S7_v = add_lf(Y[2], Y[3])
        
        inner_terms = [
            (S1_u, S1_v),
            (S2_u, S2_v),
            (S3_u, S3_v),
            (S4_u, S4_v),
            (S5_u, S5_v),
            (S6_u, S6_v),
            (S7_u, S7_v)
        ]
        
        for s_idx in range(7):
            u_lf = inner_terms[s_idx][0]
            v_lf = inner_terms[s_idx][1]
            
            w_lf = {}
            for z_idx, z_sign in z_coeffs[s_idx].items():
                for c_br, c_bc, c_sign in step["C_map"][z_idx]:
                    base = c_br*8 + c_bc*2
                    # Z entry (z_r, z_c) maps to C block (c_br, c_bc) entry (z_r, z_c)
                    # Z[0] is (0,0) -> offset 0
                    # Z[1] is (0,1) -> offset 1
                    # Z[2] is (1,0) -> offset 4
                    # Z[3] is (1,1) -> offset 5
                    offsets = [0, 1, 4, 5]
                    c_idx = base + offsets[z_idx]
                    val = z_sign * c_sign
                    w_lf[c_idx] = w_lf.get(c_idx, 0) + val
                    if w_lf[c_idx] == 0:
                        del w_lf[c_idx]
            
            # Add term
            for idx, val in u_lf.items():
                U[idx, col] += val
            for idx, val in v_lf.items():
                V[idx, col] += val
            for idx, val in w_lf.items():
                W[idx, col] += val
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
