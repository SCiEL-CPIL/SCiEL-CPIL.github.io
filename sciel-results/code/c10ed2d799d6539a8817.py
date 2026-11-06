lpf = np.zeros(N_PLUS_1 + 1, dtype=np.int32)
for i in range(2, N_PLUS_1 + 1):
    if lpf[i] == 0:
        lpf[i::i] = i
