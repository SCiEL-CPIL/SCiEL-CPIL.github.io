def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    subproducts = [
        ({'A00':0,'A01':1,'A10':4,'A11':5}, {'B00':0,'B01':1,'B10':4,'B11':5}, {'Z00':0,'Z01':2,'Z10':1,'Z11':3}),
        ({'A00':2,'A01':3,'A10':6,'A11':7}, {'B00':8,'B01':9,'B10':12,'B11':13}, {'Z00':0,'Z01':2,'Z10':1,'Z11':3}),
        ({'A00':0,'A01':1,'A10':4,'A11':5}, {'B00':2,'B01':3,'B10':6,'B11':7}, {'Z00':4,'Z01':6,'Z10':5,'Z11':7}),
        ({'A00':2,'A01':3,'A10':6,'A11':7}, {'B00':10,'B01':11,'B10':14,'B11':15}, {'Z00':4,'Z01':6,'Z10':5,'Z11':7}),
    ]
    
    strassen_terms = [
        ({'A00':1,'A11':1}, {'B00':1,'B11':1}),
        ({'A10':1,'A11':1}, {'B00':1}),
        ({'A00':1}, {'B10':1,'B11':-1}),
        ({'A11':1}, {'B01':1,'B10':-1}),
        ({'A00':1,'A01':1}, {'B11':1}),
        ({'A10':1,'A00':-1}, {'B00':1,'B10':1}),
        ({'A01':1,'A11':-1}, {'B01':1,'B11':1}),
    ]
    
    w_contrib = [
        {'Z00':1,'Z11':1},
        {'Z10':1,'Z11':-1},
        {'Z01':1,'Z11':1},
        {'Z00':1,'Z10':1},
        {'Z00':-1,'Z01':1},
        {'Z11':1},
        {'Z00':1},
    ]
    
    col = 0
    for A_idx, B_idx, W_row in subproducts:
        for t in range(7):
            u_coeffs, v_coeffs = strassen_terms[t]
            for name, coeff in u_coeffs.items():
                U[A_idx[name]][col] += coeff
            for name, coeff in v_coeffs.items():
                V[B_idx[name]][col] += coeff
            for zname, sign in w_contrib[t].items():
                W[W_row[zname]][col] += sign
            col += 1
    
    return {"U": U, "V": V, "W": W}
