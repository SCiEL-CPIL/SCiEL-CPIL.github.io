def construct():
    import itertools
    import random
    
    n = 9
    all_edges = list(itertools.combinations(range(n), 5))
    
    def is_non_2_colorable(edges):
        if not edges:
            return False
        edges_set = [set(e) for e in edges]
        for mask in range(1 << n):
            color = [(mask >> i) & 1 for i in range(n)]
            mono = False
            for e in edges_set:
                vals = [color[v] for v in e]
                if all(v == 0 for v in vals) or all(v == 1 for v in vals):
                    mono = True
                    break
            if not mono:
                return False
        return True
    
    best = list(all_edges)
    for seed in range(50):
        random.seed(seed)
        edges = list(all_edges)
        random.shuffle(edges)
        for i in range(len(edges)):
            if i >= len(edges):
                break
            trial = edges[:i] + edges[i+1:]
            if is_non_2_colorable(trial):
                edges = trial
        if len(edges) < len(best):
            best = edges
    
    return {'n': n, 'edges': [list(e) for e in best]}
