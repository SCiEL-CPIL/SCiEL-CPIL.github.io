    V4_full = np.zeros((20, 49))
    W4_full = np.zeros((20, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    # V index in full 4x5 matrix: j*5+k
                                    V4_full[j*5+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                                    # W index in full 5x4 matrix (transposed C): k*4+i
                                    W4_full[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
