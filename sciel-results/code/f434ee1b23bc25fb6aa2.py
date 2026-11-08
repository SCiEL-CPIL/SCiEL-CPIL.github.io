    for t in terms:
        if isinstance(t, tuple) and len(t) == 3:
            # Schoolbook
            u_idx, v_idx, w_idx = t
            u = {u_idx: 1}
            v = {v_idx: 1}
            w = {w_idx: 1}
            unified_terms.append((u, v, w))
        else:
            # Strassen
            u, v, w = t
            unified_terms.append((u, v, w))
