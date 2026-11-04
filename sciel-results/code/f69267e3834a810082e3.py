def greedy_remove(U, V, W, rank, order):
    U_test = U.copy()
    V_test = V.copy()
    W_test = W.copy()
    current_rank = rank
    removed = set()
    for t in order:
        if t in removed or t >= current_rank:
            continue
        # Create a copy without column t
        mask = [i for i in range(current_rank) if i != t]
        U_no_t = U_test[:, mask]
        V_no_t = V_test[:, mask]
        W_no_t = W_test[:, mask]
        if is_valid(U_no_t, V_no_t, W_no_t, current_rank - 1):
            U_test = U_no_t
            V_test = V_no_t
            W_test = W_no_t
            current_rank -= 1
            removed.add(t)
    return U_test, V_test, W_test, current_rank
