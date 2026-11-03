for t in range(7):
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                b = li * 2 + lj
                coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
