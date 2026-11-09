for i in range(n):
    for j in range(m):
        for k in range(p):
            total = Fraction(0)
            for c in range(rank):
                total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
            if total != 1:
                return False
