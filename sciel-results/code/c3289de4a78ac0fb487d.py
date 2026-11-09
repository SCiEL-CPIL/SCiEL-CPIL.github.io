def any_q3_exists(adj_bits):
    n = 16
    for subset in itertools.combinations(range(n), 8):
        mask = 0
        for i in range(8):
            vi = subset[i]
            for j in range(i+1, 8):
                vj = subset[j]
                if (adj_bits[vi] >> vj) & 1:
                    idx = get_edge_index(i, j)
                    mask |= (1 << idx)
        
        for p in q3_patterns_list:
            if (mask & p) == p:
                return True
    return False
