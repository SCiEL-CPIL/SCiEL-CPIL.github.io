    col = 0
    # Define products as (X_coeffs, Y_coeffs, [(out_bi, out_bk, sign), ...])
    
    products = []
    
    # M1
    X1 = [
        make_coeff_dict([(0,0,1), (2,2,1)]),
        make_coeff_dict([(0,1,1), (2,3,1)]),
        make_coeff_dict([(1,0,1), (3,2,1)]),
        make_coeff_dict([(1,1,1), (3,3,1)])
    ]
    Y1 = [
        make_coeff_dict([(0,0,1), (2,2,1)]),
        make_coeff_dict([(0,1,1), (2,3,1)]),
        make_coeff_dict([(1,0,1), (3,2,1)]),
        make_coeff_dict([(1,1,1), (3,3,1)])
    ]
    products.append((X1, Y1, [(0,0,1), (1,1,1)]))
    
    # M2
    X2 = [
        make_coeff_dict([(2,0,1), (2,2,1)]),
        make_coeff_dict([(2,1,1), (2,3,1)]),
        make_coeff_dict([(3,0,1), (3,2,1)]),
        make_coeff_dict([(3,1,1), (3,3,1)])
    ]
    Y2 = [
        make_coeff_dict([(0,0,1)]),
        make_coeff_dict([(0,1,1)]),
        make_coeff_dict([(1,0,1)]),
        make_coeff_dict([(1,1,1)])
    ]
    products.append((X2, Y2, [(1,0,1), (1,1,-1)]))
    
    # M3
    X3 = [
        make_coeff_dict([(0,0,1)]),
        make_coeff_dict([(0,1,1)]),
        make_coeff_dict([(1,0,1)]),
        make_coeff_dict([(1,1,1)])
    ]
    Y3 = [
        make_coeff_dict([(0,1,1), (2,1,-1)]),
        make_coeff_dict([(0,2,1), (2,2,-1)]),
        make_coeff_dict([(1,1,1), (3,1,-1)]),
        make_coeff_dict([(1,2,1), (3,2,-1)])
    ]
    products.append((X3, Y3, [(0,1,1), (1,1,1)]))
    
    # M4
    X4 = [
        make_coeff_dict([(2,2,1)]),
        make_coeff_dict([(2,3,1)]),
        make_coeff_dict([(3,2,1)]),
        make_coeff_dict([(3,3,1)])
    ]
    Y4 = [
        make_coeff_dict([(1,0,1), (0,0,-1)]),
        make_coeff_dict([(1,1,1), (0,1,-1)]),
        make_coeff_dict([(2,0,1), (1,0,-1)]),
        make_coeff_dict([(2,1,1), (1,1,-1)])
    ]
    products.append((X4, Y4, [(0,0,1), (1,0,1)]))
    
    # M5
    X5 = [
        make_coeff_dict([(0,0,1), (0,2,1)]),
        make_coeff_dict([(0,1,1), (0,3,1)]),
        make_coeff_dict([(1,0,1), (1,2,1)]),
        make_coeff_dict([(1,1,1), (1,3,1)])
    ]
    Y5 = [
        make_coeff_dict([(2,2,1)]),
        make_coeff_dict([(2,3,1)]),
        make_coeff_dict([(3,2,1)]),
        make_coeff_dict([(3,3,1)])
    ]
    products.append((X5, Y5, [(0,1,1), (0,0,-1)]))
    
    # M6
    X6 = [
        make_coeff_dict([(2,0,1), (0,0,-1)]),
        make_coeff_dict([(2,1,1), (0,1,-1)]),
        make_coeff_dict([(3,0,1), (1,0,-1)]),
        make_coeff_dict([(3,1,1), (1,1,-1)])
    ]
    Y6 = [
        make_coeff_dict([(0,0,1), (0,2,1)]),
        make_coeff_dict([(0,1,1), (0,3,1)]),
        make_coeff_dict([(1,0,1), (1,2,1)]),
        make_coeff_dict([(1,1,1), (1,3,1)])
    ]
    products.append((X6, Y6, [(1,1,1)]))
    
    # M7
    X7 = [
        make_coeff_dict([(0,2,1), (2,2,-1)]),
        make_coeff_dict([(0,3,1), (2,3,-1)]),
        make_coeff_dict([(1,2,1), (3,2,-1)]),
        make_coeff_dict([(1,3,1), (3,3,-1)])
    ]
    Y7 = [
        make_coeff_dict([(1,0,1), (2,0,1)]),
        make_coeff_dict([(1,1,1), (2,1,1)]),
        make_coeff_dict([(2,0,1), (3,0,1)]),
        make_coeff_dict([(2,1,1), (3,1,1)])
    ]
    products.append((X7, Y7, [(0,0,1)]))
    
    for X, Y, outs in products:
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    # This is one term of the 2x2 product
                    # U part: X[p,r]
                    # V part: Y[r,q]
                    
                    for (i, j), val in X[p*2 + r].items():
                        U[i * m + j][col] += val
                        
                    for (j, k), val in Y[r*2 + q].items():
                        V[j * p + k][col] += val
                        
                    # W part: update for all output blocks
                    for out_bi, out_bk, sign in outs:
                        gi = 2 * out_bi + p
                        gk = 2 * out_bk + q
                        W[gk * n + gi][col] += sign
                        
                    col += 1
                    
    # Last column
    for j in range(4):
        for i in range(4):
            U[i * m + j][col] = 1
        V[j * p + 4][col] = 1
        for i in range(4):
            W[4 * n + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
