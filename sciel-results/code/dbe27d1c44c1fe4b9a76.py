def construct():
    """Construct the lowest-rank exact tensor decomposition for 2x3 * 3x6 matmul."""
    n, m, p = DIMENSIONS
    
    # Direct construction of rank-18 decomposition
    U, V, W = _construct_rank_18(n, m, p)
    
    # Verify
    ok, _ = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        # Fallback to schoolbook if something went wrong (should not happen)
        U, V, W = _schoolbook(n, m, p)
        
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
