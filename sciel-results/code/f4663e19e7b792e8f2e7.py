<<<<<<< SEARCH
    for target_rank in range(known_rank - 1, max(known_rank - 6, 27), -1):
        if time.time() > deadline - 5:
            break
        result = _try_reduce_rank(U, V, W, target_rank, deadline - 3, n, m, p)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return known_result
=======
    for target_rank in range(known_rank - 1, max(known_rank - 8, 25), -1):
        if time.time() > deadline - 10:
            break
        result = _try_reduce_rank(U, V, W, target_rank, deadline - 5, n, m, p)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return known_result
>>>>>>> REPLACE
