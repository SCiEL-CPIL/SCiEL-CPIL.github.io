<<<<<<< SEARCH
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
            col += 1
            
    # Post-processing: Remove duplicate columns
    keep_cols = []
    for i in range(rank):
        is_dup = False
        for j in range(i):
            if j in keep_cols:
                if np.array_equal(U[:, i], U[:, j]) and np.array_equal(V[:, i], V[:, j]) and np.array_equal(W[:, i], W[:, j]):
                    # Merge: add column i to column j (which is kept)
                    # Since they are identical, U_i V_i W_i = U_j V_j W_j
                    # Sum = 2 * U_j V_j W_j
                    # So we multiply the kept column by 2
                    U[:, j] *= 2
                    V[:, j] *= 2
                    W[:, j] *= 2
                    is_dup = True
                    break
        if not is_dup:
            keep_cols.append(i)
            
    if len(keep_cols) < rank:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
