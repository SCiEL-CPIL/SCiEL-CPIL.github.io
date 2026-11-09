def add_strassen_block(U, V, W, 
                       a_idx, b_idx, c_idx, d_idx,
                       e_idx, f_idx, g_idx, h_idx,
                       y_idx, z_idx, w_idx, x_idx):
    # M1 = (a+b)(e+f)
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][a_idx] = 1; U[-1][b_idx] = 1
    V[-1][e_idx] = 1; V[-1][f_idx] = 1
    W[-1][y_idx] = 1; W[-1][x_idx] = 1
    
    # M2 = (c+d)e
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][c_idx] = 1; U[-1][d_idx] = 1
    V[-1][e_idx] = 1
    W[-1][w_idx] = 1
    
    # M3 = (a-b)(g+h)
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][a_idx] = 1; U[-1][b_idx] = -1
    V[-1][g_idx] = 1; V[-1][h_idx] = 1
    W[-1][z_idx] = 1
    
    # M4 = d(h)
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][d_idx] = 1
    V[-1][h_idx] = 1
    W[-1][y_idx] = 1; W[-1][x_idx] = 1
    
    # M5 = (a+d)(e+h)
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][a_idx] = 1; U[-1][d_idx] = 1
    V[-1][e_idx] = 1; V[-1][h_idx] = 1
    W[-1][y_idx] = -1; W[-1][w_idx] = 1
    
    # M6 = (b+c)h
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][b_idx] = 1; U[-1][c_idx] = 1
    V[-1][h_idx] = 1
    W[-1][x_idx] = 1
    
    # M7 = (b-c)(e+g)
    U.append([0]*12); V.append([0]*16); W.append([0]*12)
    U[-1][b_idx] = 1; U[-1][c_idx] = -1
    V[-1][e_idx] = 1; V[-1][g_idx] = 1
    W[-1][y_idx] = 1
