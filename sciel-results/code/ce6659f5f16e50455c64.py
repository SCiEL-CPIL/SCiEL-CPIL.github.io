import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        rng = Random(seed)
        num_i = n - k
        adj_K = [set() for _ in range(num_i)]
        for i in range(num_i):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_K[i].add(u)
        
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
            
        for i in range(num_i):
            v = k + i
            for u in adj_K[i]:
                cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
