def construct():
    n, m, p = 5, 5, 5
    
    # Build 4x4 Strassen
    U4, V4, W4 = build_4x4_strassen()
    
    # Extend to 5x5
    rank = 49 + 61
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    # Copy 4x4 part
    U[:16, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    col = 49
    
    # Add all 61 cross terms
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue  # Already handled by 4x4 Strassen
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*5+i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
