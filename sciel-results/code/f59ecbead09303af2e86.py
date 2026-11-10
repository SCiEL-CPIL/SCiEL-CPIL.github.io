for v in range(k, n):
    for u in range(k):
        if rng.random() < p:
            cross_neighbors[v].append(u)
