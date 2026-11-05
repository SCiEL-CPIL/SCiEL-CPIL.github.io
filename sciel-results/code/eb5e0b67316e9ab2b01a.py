for S in range(2, MOD - 1):
    E = S - 1
    while E < MOD - 1:
        new_E = E + 1
        if new_E == 1008:
            break
        if S <= inv[new_E] <= E:
            break
        E = new_E
    if E >= S:
        current_len = E - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S
