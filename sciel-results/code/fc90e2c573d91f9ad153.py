    def add_1x4x4(i_base, j_base, k_base):
        nonlocal col
        for lj in range(4):
            U[i_base * m + (j_base + lj), col] = 1.0
            for lk in range(4):
                V[(j_base + lj) * p + (k_base + lk), col] = 1.0
        for lk in range(4):
            W[(k_base + lk) * n + i_base, col] = 1.0
        col += 1
