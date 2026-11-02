<<<<<<< SEARCH
    # Build and verify the rank-25 decomposition
    result = build_rank25()
    
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
=======
    # Try rank-24 perturbation first
    result_24 = build_rank24_perturbed()
    if result_24 is not None and verify_decomp(result_24["U"], result_24["V"], result_24["W"]):
        return result_24
    
    # Build and verify the rank-25 decomposition
    result = build_rank25()
    
    if verify_decomp(result["U"], result["V"], result["W"]):
        return result

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
>>>>>>> REPLACE
