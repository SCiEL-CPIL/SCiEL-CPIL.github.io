C_decomp = [[0 for _ in range(p)] for _ in range(n)]
for i in range(n):
    for k in range(p):
        val = 0
        for r in range(rank):
            ua = 0
            for jj in range(m):
                ua += result['U'][i * m + jj][r] * A[i][jj]
            vb = 0
            for jj in range(m):
                vb += result['V'][jj * p + k][r] * B[jj][k]
            val += result['W'][k * n + i][r] * ua * vb
        C_decomp[i][k] = val
