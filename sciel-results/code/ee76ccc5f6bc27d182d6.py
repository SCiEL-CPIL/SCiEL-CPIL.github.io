<<<<<<< SEARCH
def construct():
    start_time = time.time()

    # Try rank 22 first (two 2x3x3 blocks)
    best = _build_rank22()
    if _verify(best["U"], best["V"], best["W"]):
        return best

    # Fallback to rank 33 (three 2x3x2 blocks)
    best = _build_rank33()
    if _verify(best["U"], best["V"], best["W"]):
        return best

    # Should not reach here, but return best effort
    return best
=======
def construct():
    start_time = time.time()

    # Try rank 33 (three 2x3x2 blocks) as it is more likely correct than the flawed rank 22
    best = _build_rank33()
    if _verify(best["U"], best["V"], best["W"]):
        return {"U": np.array(best["U"], dtype=float), "V": np.array(best["V"], dtype=float), "W": np.array(best["W"], dtype=float)}

    # Fallback to rank 22 (two 2x3x3 blocks) - might be invalid but try anyway
    best = _build_rank22()
    if _verify(best["U"], best["V"], best["W"]):
        return {"U": np.array(best["U"], dtype=float), "V": np.array(best["V"], dtype=float), "W": np.array(best["W"], dtype=float)}

    # Fallback to trivial rank 36
    best = _build_rank36()
    return {"U": np.array(best["U"], dtype=float), "V": np.array(best["V"], dtype=float), "W": np.array(best["W"], dtype=float)}

def _build_rank36():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
