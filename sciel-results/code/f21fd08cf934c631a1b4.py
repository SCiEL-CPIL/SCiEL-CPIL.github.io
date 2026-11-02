<<<<<<< SEARCH
    # Initialize: length-1 walks: for candidate c, pairs are [(j, S_c[j]) for j in range(k)]
    # Store as list of arrays per candidate
    cand_states = []
    for c in range(n_cand):
        q, S_list = candidates[c]
        k = len(S_list)
        S_arr = np.array(S_list, dtype=np.int64)
        neg_idx = np.array([S_list.index((-x) % q) for x in S_list], dtype=np.int64)
        last_arr = np.arange(k, dtype=np.int64)
        sum_arr = S_arr.copy()
        cand_states.append((q, S_arr, neg_idx, last_arr, sum_arr))
=======
    # Initialize: length-1 walks: for candidate c, pairs are [(j, S_c[j]) for j in range(k)]
    # Store as list of arrays per candidate
    cand_states = []
    for c in range(n_cand):
        q, S_list = candidates[c]
        k = len(S_list)
        S_arr = np.array(S_list, dtype=np.int64)
        # Build neg_idx: for each position j, find position where S_list[pos] == -S_list[j] mod q
        S_neg_map = {x: (-x) % q for x in S_list}
        S_inv = {v: i for i, v in enumerate(S_list)}
        neg_idx = np.array([S_inv[S_neg_map[S_list[j]]] for j in range(k)], dtype=np.int64)
        last_arr = np.arange(k, dtype=np.int64)
        sum_arr = S_arr.copy()
        cand_states.append((q, S_arr, neg_idx, last_arr, sum_arr))
>>>>>>> REPLACE
