    result = []
    for i in range(n_edges_total):
        if (best_mask >> i) & 1:
            result.append(list(edges[i]))
    return result
    