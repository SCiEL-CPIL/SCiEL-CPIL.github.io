    for c in range(7):
        for i in range(4):
            U[i][7+c] = U_str[i][c]
        V[2][7+c] = V_str[0][c]
        V[3][7+c] = V_str[1][c]
        V[9][7+c] = V_str[2][c]
        V[10][7+c] = V_str[3][c]
        W[4][7+c] = W_str[0][c]
        W[5][7+c] = W_str[1][c]
        W[6][7+c] = W_str[2][c]
        W[7][7+c] = W_str[3][c]
