# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Use Rank 92 decomposition:
    # Partition 6x6 into 3x3 blocks of 2x2 matrices.
    # Use Winograd's algorithm (Rank 23) for the 3x3 block structure.
    # Use Schoolbook (Rank 4) for each 2x2 block multiplication.
    # Total Rank = 23 * 4 = 92.
    
    rank = 92
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 92)
    V = [[0] * rank for _ in range(m * p)]  # (36, 92)
    W = [[0] * rank for _ in range(p * n)]  # (36, 92)
    
    # Winograd 3x3 decomposition terms
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 9 entries (row-major) for A
    # v_coeffs: 9 entries (row-major) for B
    # w_coeffs: 9 entries (row-major) for C
    
    # The 23 terms of Winograd's 3x3 algorithm
    # Note: The exact coefficients vary by implementation, but rank is 23.
    # Here we use a known set.
    
    winograd_3x3_terms = [
        # 1
        ([1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1], [1,0,0,0,1,0,0,0,1]),
        # 2
        ([1,0,0,0,-1,0,0,0,-1], [1,0,0,0,1,0,0,0,1], [0,0,1,0,0,0,1,0,0]),
        # 3
        ([0,1,0,0,0,-1,0,0,-1], [1,0,0,0,1,0,0,0,1], [0,1,0,0,0,0,0,0,1]),
        # 4
        ([0,0,1,0,0,0,-1,0,1], [1,0,0,0,1,0,0,0,1], [0,0,0,1,0,0,0,0,0]),
        # 5
        ([1,0,0,0,0,0,0,0,-1], [1,0,0,0,0,0,0,0,1], [0,0,0,0,0,1,0,0,0]),
        # 6
        ([0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
        # 7
        ([0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
        # 8
        ([0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 9
        ([0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        # 10
        ([0,0,0,0,0,1,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]),
        # 11
        ([0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]),
        # 12
        ([0,0,0,0,0,0,0,1,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0]),
        # 13
        ([0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
        # 14
        ([1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        # 15
        ([0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]),
        # 16
        ([0,0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
        # 17
        ([0,0,0,1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        # 18
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0,0]),
        # 19
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0,0]),
        # 20
        ([0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0,0]),
        # 21
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0]),
        # 22
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]), # Dummy
        # 23
        ([0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]), # Dummy
    ]
    
    # This hardcoded list is likely incorrect or incomplete for a true rank 23.
    # A true rank 23 decomposition is non-trivial.
    # I will use a safer approach: 2x2 Strassen (7) with 3x3 Schoolbook (27) is 189.
    # I will stick to the current 189 if I can't guarantee 92.
    
    # However, I can try to improve the code to use numpy for faster construction?
    # The score is based on rank, not speed.
    
    # Let's try to use a known valid decomposition for 6x6 with rank < 189.
    # Rank 112 is known.
    # But 92 is better.
    
    # I will return the current 189 decomposition to be safe, as I cannot verify the Winograd coefficients.
    
    # REVERT TO CURRENT LOGIC BUT CLEANED UP
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    c_contributions = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bi + r
                        gc = 2 * bk + c
                        U[gr * 6 + gc][col] += coeff
                    
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bk + r
                        gc = 2 * bj + c
                        V[gr * 6 + gc][col] += coeff
                    
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        gr = 2 * bj + c
                        gc = 2 * bi + r
                        W[gr * 6 + gc][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
