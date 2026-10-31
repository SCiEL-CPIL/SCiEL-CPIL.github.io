"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
_DEADLINE = time.time() + 145  # 145s safety margin before 150s kill

def _strassen_2x2():
    """Return the 7 Strassen rank-1 terms for 2x2 matrix multiplication.
    A=[a,b,c,d], B=[e,f,g,h], C=A*B
    M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    """
    return [
        ([1,0,0,1], [1,1,0,0], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # M3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # M7
    ]

def construct():
    n, m, p = DIMENSIONS
    s2_terms = _strassen_2x2()
    
    # Block-level Strassen: 7 products
    # S1=(A00+A11)(B00+B11), S2=(A10+A11)B00, S3=A00(B01-B11),
    # S4=A11(B10-B00), S5=(A00+A01)B11, S6=(A10-A00)(B00+B01), S7=(A01-A11)(B10+B11)
    # C00=S1+S4-S5+S7, C01=S3+S5, C10=S2+S4, C11=S1-S2+S3+S6
    block_defs = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for j in range(7):
        A_blocks, B_blocks, C_blocks = block_defs[j]
        for (u_inner, v_inner, w_inner) in s2_terms:
            for (block_idx, coeff_a) in A_blocks:
                br, bc = block_idx
                for pos in range(4):
                    bi, bj = pos // 2, pos % 2
                    row = 2*br + bi
                    col_idx = 2*bc + bj
                    U[row*4 + col_idx][col] += coeff_a * u_inner[pos]
            for (block_idx, coeff_b) in B_blocks:
                br, bc = block_idx
                for pos in range(4):
                    bi, bj = pos // 2, pos % 2
                    row = 2*br + bi
                    col_idx = 2*bc + bj
                    V[row*4 + col_idx][col] += coeff_b * v_inner[pos]
            for (block_idx, coeff_c) in C_blocks:
                br, bc = block_idx
                for pos in range(4):
                    bi, bj = pos // 2, pos % 2
                    row = 2*br + bi
                    col_idx = 2*bc + bj
                    W[row*4 + col_idx][col] += coeff_c * w_inner[pos]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
