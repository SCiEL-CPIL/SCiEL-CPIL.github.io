        # Diagonal blocks i == i
        # H_ii = sum_{k != i} ( 2/d_ik^4 * u_ik u_ik^T - 1/d_ik^2 * I )
        for i in range(n):
            block_diag = np.zeros((3, 3))
            for j in range(n):
                if i == j: continue
                u = pts[i] - pts[j]
                d2 = np.dot(u, u)
                if d2 < 1e-18: continue
                d4 = d2 * d2
                inv_d2 = 1.0 / d2
                inv_d4 = 1.0 / d4
                block_diag += 2.0 * inv_d4 * np.outer(u, u) - inv_d2 * np.eye(3)
            H[3*i:3*i+3, 3*i:3*i+3] = block_diag
