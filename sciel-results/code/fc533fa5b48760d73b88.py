# Part 2: 4x4 x 4x3 -> 4x3 (k=4..6) using schoolbook
c_offset = 49  # Start after the Strassen part
for j in range(4):
    for i in range(4):
        for k in range(4, 7):
            c_index = c_offset
            c_offset += 1
            U[i * 4 + j][c_index] = 1
            V[j * 7 + k][c_index] = 1
            W[k * 4 + i][c_index] = 1
