import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    # Strassen rank for 6x6 using 3x3 blocks
    # 7 multiplications * 27 scalar mults = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen multipliers
    # Each entry is (X_blocks, Y_blocks, Out_blocks)
    # blocks are indices 0..3
    # 0: TL, 1: TR, 2: BL, 3: BR
    multipliers = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),      # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),             # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),             # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),             # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),             # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),             # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])              # M7
    ]
    
    col = 0
    for x_blocks, y_blocks, out_blocks in multipliers:
        for a in range(3):
            for s in range(3):
                for b in range(3):
                    if time.time() > deadline:
                        break
                    # X part
                    for block, sign in x_blocks:
                        i = (block // 2) * 3 + a
                        j = (block % 2) * 3 + s
                        U[i * m + j][col] += sign
                    
                    # Y part
                    for block, sign in y_blocks:
                        j = (block // 2) * 3 + s
                        k = (block % 2) * 3 + b
                        V[j * p + k][col] += sign
                    
                    # W part
                    for block, sign in out_blocks:
                        i = (block // 2) * 3 + a
                        k = (block % 2) * 3 + b
                        W[k * n + i][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
