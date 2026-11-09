<<<<<<< SEARCH
    # Verify correctness within time budget (vectorized)
    if time.time() < deadline - 5:
        all_ok = True
        for test_i in range(n):
            for test_j in range(m):
                u_row = U[test_i * m + test_j, :]
                for test_k in range(p):
                    v_row = V[test_j * p + test_k, :]
                    w_row = W[test_k * n + test_i, :]
                    actual = np.dot(u_row, v_row * w_row)
                    if actual != 1:
                        all_ok = False
                        break
                if not all_ok:
                    break
            if not all_ok:
                break
        
        if not all_ok:
            # Fallback: return standard rank-100 decomposition
            rank_std = n * m * p
            U_std = np.zeros((nm, rank_std), dtype=np.int64)
            V_std = np.zeros((mp, rank_std), dtype=np.int64)
            W_std = np.zeros((pn, rank_std), dtype=np.int64)
            col_std = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_std[i * m + j, col_std] = 1
                        V_std[j * p + k, col_std] = 1
                        W_std[k * n + i, col_std] = 1
                        col_std += 1
            return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # Verify correctness within time budget
    if time.time() < deadline - 5:
        all_ok = True
        for test_i in range(n):
            for test_j in range(m):
                u_row = U[test_i * m + test_j, :]
                for test_k in range(p):
                    v_row = V[test_j * p + test_k, :]
                    w_row = W[test_k * n + test_i, :]
                    actual = np.dot(u_row, v_row * w_row)
                    if actual != 1:
                        all_ok = False
                        break
                if not all_ok:
                    break
            if not all_ok:
                break
        
        if not all_ok:
            # Fallback: return standard rank-100 decomposition
            rank_std = n * m * p
            U_std = np.zeros((nm, rank_std), dtype=np.int64)
            V_std = np.zeros((mp, rank_std), dtype=np.int64)
            W_std = np.zeros((pn, rank_std), dtype=np.int64)
            col_std = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_std[i * m + j, col_std] = 1
                        V_std[j * p + k, col_std] = 1
                        W_std[k * n + i, col_std] = 1
                        col_std += 1
            return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
