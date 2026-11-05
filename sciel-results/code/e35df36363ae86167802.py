"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n = m = p = 6
    R = 189  # 7 Strassen terms * 27 schoolbook scalar mults per 3x3 product
    U = [[0] * R for _ in range(36)]
    V = [[0] * R for _ in range(36)]
    W = [[0] * R for _ in range(36)]
    
    # Level-1 Strassen: split 6x6 into 2x2 blocks of 3x3
    # Term definitions: (left_blocks_with_coeffs, right_blocks_with_coeffs)
    # Block (br,bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),  # (A00+A11)*(B00+B11)
        ({(1,0):1,(1,1):1}, {(0,0):1}),           # (A10+A11)*B00
        ({(0,0):1}, {(0,1):1,(1,1):-1}),          # A00*(B01-B11)
        ({(1,1):1}, {(1,0):1,(0,0):-1}),          # A11*(B10-B00)
        ({(0,0):1,(0,1):1}, {(1,1):1}),           # (A00+A01)*B11
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),  # (A10-A00)*(B00+B01)
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),  # (A01-A11)*(B10+B11)
    ]
    
    # How each term contributes to result blocks: (block_row, block_col): sign
    contribs = [
        {(0,0):1,(1,1):1},       # T1 -> C00(+1), C11(+1)
        {(1,0):1,(1,1):-1},      # T2 -> C10(+1), C11(-1)
        {(0,1):1,(1,1):1},       # T3 -> C01(+1), C11(+1)
        {(0,0):1,(1,0):1},       # T4 -> C00(+1), C10(+1)
        {(0,0):-1,(0,1):1},      # T5 -> C00(-1), C01(+1)
        {(1,1):1},               # T6 -> C11(+1)
        {(0,0):1},               # T7 -> C00(+1)
    ]
    
    idx = 0
    for t in range(7):
        L, Rr = terms[t]
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U: left factor coefficients
                    for (br, bc), c in L.items():
                        ar = 3*br + i
                        ac = 3*bc + a
                        U[ar*m + ac][idx] += c
                    # V: right factor coefficients
                    for (br, bc), c in Rr.items():
                        br_ = 3*br + a
                        bc_ = 3*bc + k
                        V[br_*p + bc_][idx] += c
                    # W: output contributions
                    for (rbr, rbc), s in contribs[t].items():
                        ci = 3*rbr + i
                        ck = 3*rbc + k
                        W[ck*n + ci][idx] += s
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
