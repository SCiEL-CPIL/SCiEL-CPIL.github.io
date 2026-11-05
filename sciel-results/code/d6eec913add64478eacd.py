# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        m = len(edges)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                        u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def greedy_induced_matching(conflict, remaining, rng, heuristic='min'):
        current_set = set()
        active = set(remaining)
        while active:
            if heuristic == 'min':
                chosen = min(active, key=lambda v: len(conflict[v] & active))
            elif heuristic == 'max':
                chosen = max(active, key=lambda v: len(conflict[v] & active))
            else:
                chosen = rng.choice(list(active))
            current_set.add(chosen)
            active.discard(chosen)
            for nb in conflict[chosen]:
                active.discard(nb)
        return current_set

    def local_search(conflict, remaining, initial_set, rng, max_iters=100):
        current_set = set(initial_set)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1-opt: add a vertex
            for v in remaining:
                if v not in current_set and not (conflict[v] & current_set):
                    current_set.add(v)
                    improved = True
            
            # 2-opt: remove one, add multiple
            for s in list(current_set):
                temp_set = current_set - {s}
                candidates = [v for v in remaining if v not in temp_set and not (conflict[v] & temp_set)]
                if not candidates:
                    continue
                
                cand_active = set(candidates)
                cand_set = set()
                while cand_active:
                    chosen = min(cand_active, key=lambda v: len(conflict[v] & cand_active))
                    cand_set.add(chosen)
                    cand_active.discard(chosen)
                    for nb in conflict[chosen]:
                        cand_active.discard(nb)
                
                if len(cand_set) > 1:
                    new_set = temp_set | cand_set
                    if len(new_set) > len(current_set):
                        current_set = new_set
                        improved = True
                        break
        
        return current_set

    def find_max_induced_matching(conflict, remaining, rng):
        best_set = set()
        
        for heuristic in ['min', 'max', 'random']:
            for restart in range(5):
                init = greedy_induced_matching(conflict, remaining, rng, heuristic)
                improved = local_search(conflict, remaining, init, rng, max_iters=50)
                if len(improved) > len(best_set):
                    best_set = improved
        
        return best_set

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_graph(edges, adj)
        
        color = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        rng = random.Random(seed + 999)
        
        while remaining:
            selected = find_max_induced_matching(conflict, remaining, rng)
            if not selected:
                selected = {next(iter(remaining))}
            for idx in selected:
                color[idx] = color_idx
                remaining.discard(idx)
            color_idx += 1
        
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
