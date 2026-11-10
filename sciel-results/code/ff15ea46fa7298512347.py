def _extend_L_from(count, start_L, L_MAX):
    L = start_L
    while L < L_MAX and count[L] >= 2:
        L += 1
    return L
