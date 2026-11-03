"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_comb(d1, d2, sign=1):
        result = dict(d1)
        for key, val in d2.items():
            result[key] = result.get(key, 0) + sign * val
        return {k: v for k, v in result.items() if abs(v) > 1e-10}
    
    def sub_comb(d1, d2):
        return add_comb(d1, d2, -1)
    
    def add_2x2(A, B, sign=1):
        result = [[None]*2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                result[i][j] = add_comb(A[i][j], B[i][j], sign)
        return result
    
    def sub_2x2(A, B):
        return add_2x2(A, B, -1)
    
    def strassen_2x2(a_coeffs, b_coeffs):
        a00, a01 = a_coeffs[0][0], a_coeffs[0][1]
        a10, a11 = a_coeffs[1][0], a_coeffs[1][1]
        b00, b01 = b_coeffs[0][0], b_coeffs[0][1]
        b10, b11 = b_coeffs[1][0], b_coeffs[1][1]
        
        products = []
        products.append((add_comb(a00, a11), add_comb(b00, b11), {(0,0): 1, (1,1): 1}))
        products.append((add_comb(a10, a11), b00, {(1,0): 1, (1,1): -1}))
        products.append((a00, sub_comb(b01, b11), {(0,1): 1, (1,1): 1}))
        products.append((a11, sub_comb(b10, b00), {(0,0): 1, (1,0): 1}))
        products.append((add_comb(a00, a01), b11, {(0,0): -1, (0,1): 1}))
        products.append((sub_comb(a10, a00), add_comb(b00, b01), {(1,1): 1}))
        products.append((sub_comb(a01, a11), add_comb(b10, b11), {(0,0): 1}))
        return products
    
    # Build A and B 2x2 block structures for 4x4*4x4 (B cols 0-3)
    A_blocks = [[None]*2 for _ in range(2)]
    B_blocks = [[None]*2 for _ in range(2)]
    for bi in range(2):
        for bj in range(2):
            A_blocks[bi][bj] = [[{(2*bi+ri, 2*bj+cj): 1.0} for cj in range(2)] for ri in range(2)]
            B_blocks[bi][bj] = [[{(2*bi+ri, 2*bj+cj): 1.0} for cj in range(2)] for ri in range(2)]
    
    X1 = add_2x2(A_blocks[0][0], A_blocks[1][1])
    Y1 = add_2x2(B_blocks[0][0], B_blocks[1][1])
    X2 = add_2x2(A_blocks[1][0], A_blocks[1][1])
    Y2 = B_blocks[0][0]
    X3 = A_blocks[0][0]
    Y3 = sub_2x2(B_blocks[0][1], B_blocks[1][1])
    X4 = A_blocks[1][1]
    Y4 = sub_2x2(B_blocks[1][0], B_blocks[0][0])
    X5 = add_2x2(A_blocks[0][0], A_blocks[0][1])
    Y5 = B_blocks[1][1]
    X6 = sub_2x2(A_blocks[1][0], A_blocks[0][0])
    Y6 = add_2x2(B_blocks[0][0], B_blocks[0][1])
    X7 = sub_2x2(A_blocks[0][1], A_blocks[1][1])
    Y7 = add_2x2(B_blocks[1][0], B_blocks[1][1])
    
    outer_products = [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]
    
    col = 0
    for X, Y in outer_products:
        inner_terms = strassen_2x2(X, Y)
        for A_dict, B_dict, out_contrib in inner_terms:
            if not A_dict or not B_dict:
                continue
            for (ai, aj), coeff in A_dict.items():
                U[ai * m + aj][col] = coeff
            for (bj, bk), coeff in B_dict.items():
                V[bj * p + bk][col] = coeff
            for (ci, ck), coeff in out_contrib.items():
                W[ck * n + ci][col] += coeff
            col += 1
    
    # Add 16 schoolbook terms for A(4x4) * B[:,4](4x1)
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1.0
            V[j * p + 4][col] = 1.0
            W[4 * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

