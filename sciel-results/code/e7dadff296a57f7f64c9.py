    x = np.arange(1, P)
    r = (x * x) % P
    qr_mask = np.zeros(P, dtype=bool)
    qr_mask[r] = True
    