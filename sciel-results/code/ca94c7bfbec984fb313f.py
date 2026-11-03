import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define inner Strassen steps
    # Each step is (X_comb, Y_comb, Z_comb)
    # X_comb: list of (r, c, sign)
    # Y_comb: list of (r, c, sign)
    # Z_comb: list of (r, c, sign)
    inner_steps = [
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (1,1,-1))]),
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1), (1,1,1))]),
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (0,1,1))]),
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1), (1,1,1))])
    ]
    
    # Outer Strassen steps
    # Each step is (X_comb, Y_comb, Z_comb)
    # X_comb: list of (br, bc, sign)
    # Y_comb: list of (br, bc, sign)
    # Z_comb: list of (br, bc, sign)
    outer_steps = [
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (1,1,-1))]),
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1), (1,1,1))]),
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (0,1,1))]),
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1), (1,1,1))])
    ]
    
    # Generate the 49 multiplications
    mults = []
    for ox, oy, oz in outer_steps:
        for ix, iy, iz in inner_steps:
            # Compose X_comb
            x_comb = []
            for (br, bc, s1) in ox:
                for (r, c, s2) in ix:
                    x_comb.append((br*2+r, bc*2+c, s1*s2))
            # Compose Y_comb
            y_comb = []
            for (br, bc, s1) in oy:
                for (r, c, s2) in iy:
                    y_comb.append((br*2+r, bc*2+c, s1*s2))
            # Compose Z_comb
            z_comb = []
            for (br, bc, s1) in oz:
                for (r, c, s2) in iz:
                    z_comb.append((br*2+r, bc*2+c, s1*s2))
            mults.append((x_comb, y_comb, z_comb))
            
    # Fill U, V, W for B_L
    for i, (x_comb, y_comb, z_comb) in enumerate(mults):
        for (r, c, s) in x_comb:
            U[r*4+c][i] += s
        for (r, c, s) in y_comb:
            V[r*8+c][i] += s
        for (r, c, s) in z_comb:
            W[c*4+r][i] += s
            
    # Fill U, V, W for B_R
    for i, (x_comb, y_comb, z_comb) in enumerate(mults):
        idx = i + 49
        for (r, c, s) in x_comb:
            U[r*4+c][idx] += s
        for (r, c, s) in y_comb:
            V[r*8+(c+4)][idx] += s
        for (r, c, s) in z_comb:
            W[(c+4)*4+r][idx] += s
            
    return {"U": U, "V": V, "W": W}
