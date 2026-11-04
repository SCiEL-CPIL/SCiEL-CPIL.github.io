def get_2x2_strassen_terms():
    # Returns 7 lists of (u, v, w) where u, v, w are lists of (i,j, coeff)
    terms = []
    # M1
    u = [(0,0,1), (1,1,1)]
    v = [(0,0,1), (1,1,1)]
    w = [(0,0,1), (1,1,1)] # C11, C22
    terms.append((u, v, w))
    # ... and so on
    return terms
