    for q in [3, 4, 5, 7, 9, 11, 13]:
        if time.time() > deadline - 20:
            break
        try:
            result = _construct_polarity_graph(q)
            # ...
            if not _verify_c4_free(n, adj):
                continue
            if _independent_set_ge(adj, n, 11, deadline - 10):
                continue
            if n > best_n:
                best_n = n
                best_edges = result['edges']
        except Exception:
            continue
    