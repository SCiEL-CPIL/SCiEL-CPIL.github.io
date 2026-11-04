    def verify_6x6(U, V, W):
        n, m, p = 6, 6, 6
        rank = W.shape[1]
        # Sample some entries to verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
                    if total != 1:
                        return False
        return True
