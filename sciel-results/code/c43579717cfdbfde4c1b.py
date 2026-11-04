        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    expected = 1
                    actual = 0.0
                    for r in range(rank_):
                        actual += U[i * m_ + j][r] * V[j * p_ + k][r] * W[k * n_ + i][r]
                    if abs(actual - expected) > tol:
                        return False
