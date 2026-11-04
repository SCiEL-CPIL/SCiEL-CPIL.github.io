            for k in range(3):
                for i in range(3):
                    Bj = Ur[i * 3 + :, :] * Vr[:, k].reshape(3, 1) * np.eye(r)[None, :]
                    # B[k,i,j,r] = Ur[i*3+j, r] * Vr[j*3+k, r]
                    for j in range(3):
                        B[k, i, j, :] = Ur[i * 3 + j, :] * Vr[j * 3 + k, :]
                    Bj = B[k, i]
    