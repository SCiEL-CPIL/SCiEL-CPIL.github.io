<<<<<<< SEARCH
    rank = len(U_list)
    U = [row for row in U_list]
    V = [row for row in V_list]
    W = [row for row in W_list]
    
    return {"U": U, "V": V, "W": W}
=======
    rank = len(U_list)
    # U_list has `rank` rows, each of length n*m=16. Need to transpose to (n*m, rank).
    U = [[U_list[c][r] for c in range(rank)] for r in range(16)]
    V = [[V_list[c][r] for c in range(rank)] for r in range(28)]
    W = [[W_list[c][r] for c in range(rank)] for r in range(28)]
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
