def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        # cross edges: for v in independent set [k, n), for u in [0,k), edge if rng.random() < p
        # Must match seed generator exactly: edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return cross
    result = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cross = instance(n, k, p, seed)
        cliques = [list(range(k))]  # the big clique on first k vertices
        for (u, v) in cross:
            cliques.append([int(u), int(v)])
        result.append(cliques)
    return result
