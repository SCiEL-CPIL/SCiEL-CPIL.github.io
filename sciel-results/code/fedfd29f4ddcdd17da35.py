"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4

    # We decompose M_{3,4,4} by splitting the middle dimension into two blocks of size 2:
    # M_{3,4,4} = M_{3,2,4} + M_{3,2,4} (with shifted indices)
    # Each M_{3,2,4} can be further split: M_{3,2,4} = M_{3,2,2} + M_{3,2,2}
    # So we need 4 copies of M_{3,2,2}.
    #
    # M_{3,2,2} = 3 independent M_{1,2,2} (one for each row i).
    # Rank of M_{1,2,2} is 3.
    # So rank of M_{3,2,2} is 3 * 3 = 9.
    # Total rank = 4 * 9 = 36.
    #
    # Rank-3 decomposition of M_{1,2,2}:
    # Inputs: A = [a0, a1], B = [b00, b01; b10, b11] (2x2)
    # Outputs: C = [c0, c1] where c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    #
    # Standard rank-3 decomposition:
    # Term 1: u=1, v=b00+b10, w=c0
    # Term 2: u=1, v=b01+b11, w=c1
    # Term 3: u=a1, v=b10-b00, w=c0  -- this doesn't work directly
    #
    # Let me use the known rank-3 decomposition for M_{1,2,2}:
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    #
    # Use:
    # t1: (a0) * (b00) -> contributes to c0
    # t2: (a0) * (b01) -> contributes to c1
    # t3: (a1) * (b10) -> contributes to c0
    # t4: (a1) * (b11) -> contributes to c1
    # That's rank 4. To get rank 3, we need a non-trivial decomposition.
    #
    # Known rank-3 for M_{1,2,2} (from literature):
    # Let's use:
    # t1: u = a0, v = b00 + b10, w = c0
    # t2: u = a0, v = b01 + b11, w = c1
    # t3: u = a1, v = b10 - b00, w = c0  -- no, this gives a1*b10 - a1*b00
    #
    # Actually, the rank of M_{1,2,2} is indeed 3. Here's a valid decomposition:
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    #
    # Define:
    # t1: u = a0, v = b00, w = c0
    # t2: u = a0, v = b01, w = c1
    # t3: u = a1, v = b10 + b11, w = c0 + c1  -- no
    #
    # Let me just use rank 4 for M_{1,2,2} to be safe, giving total rank = 4*3*4 = 48.
    # But that's the same as the seed. Let me try to get rank 3.
    #
    # After careful thought, here's a valid rank-3 decomposition of M_{1,2,2}:
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    #
    # t1: u = 1, v = b00 + b10, w = c0  -- but u should involve a0 or a1
    #
    # I'll use the trivial rank-4 for safety. Total rank = 48.
    # Actually, let me just return the naive rank-48 decomposition.

    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
