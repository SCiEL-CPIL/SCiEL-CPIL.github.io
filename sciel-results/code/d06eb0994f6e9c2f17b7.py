                def _verify_decomposition(U, V, W, n, m, p):
                    ...
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for c in range(rank):
                                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                                expected = 1.0 if i == k else 0.0
                                if abs(total - expected) > 1e-10:
                                    return False
                    return True
                