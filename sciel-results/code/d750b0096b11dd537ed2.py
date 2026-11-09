        # M7 = (A12 - A22)(B21 + B22)
        U[i0 * m + j1_idx][col] = 1
        U[i1_idx * m + j1_idx][col] = -1
        V[j1_idx * p + k0][col] = 1
        V[j1_idx * p + k1_idx][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1_idx * n + i1_idx][col] = 1
        col += 1
