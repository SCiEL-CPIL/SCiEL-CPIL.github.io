    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = U4[i * 4 + j]
                v_row = V4[j * 4 + k]
                w_row = W4[k * 4 + i]
                prod = np.sum(u_row * v_row * w_row)
                if prod != 1:
                    valid = False
                    break
