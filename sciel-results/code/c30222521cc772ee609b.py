def add_1x2x2_block(u_rows, v_rows, w_rows, offset):
    for i in range(2):
        for c in range(4):
            U[u_rows[i]][offset + c] += U1[i][c]
            V[v_rows[i]][offset + c] += V1[i][c]
            W[w_rows[i]][offset + c] += W1[i][c]
    for i in range(2, 4):
        for c in range(4):
            V[v_rows[i]][offset + c] += V1[i][c]
