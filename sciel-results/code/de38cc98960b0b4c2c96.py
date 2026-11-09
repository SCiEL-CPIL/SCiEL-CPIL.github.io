"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def build_adjacency(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj
    
    def decompose_paths(edges):
        """Decompose edges into edge-disjoint paths using greedy longest-path strategy."""
        edges_set = set(tuple(sorted(e)) for e in edges)
        adj = build_adjacency(edges)
        
        paths = []
        while edges_set:
            # Try each remaining edge as a potential start, pick the one leading to longest path
            best_path = None
            best_len = -1
            
            for start_e in list(edges_set)[:20]:  # Limit candidates for speed
                su, sv = start_e
                # Build path greedily from this edge
                path = [su, sv]
                used = {start_e}
                
                # Extend from both ends
                changed = True
                while changed:
                    changed = False
                    # Extend from end
                    end = path[-1]
                    for nb in sorted(adj.get(end, []), key=lambda x: len(adj.get(x, set()))):
                        e = (min(end, nb), max(end, nb))
                        if e not in used and nb not in path:
                            path.append(nb)
                            used.add(e)
                            changed = True
                            break
                    # Extend from start
                    st = path[0]
                    for nb in sorted(adj.get(st, []), key=lambda x: len(adj.get(x, set()))):
                        e = (min(st, nb), max(st, nb))
                        if e not in used and nb not in path:
                            path.insert(0, nb)
                            used.add(e)
                            changed = True
                            break
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None:
                e = next(iter(edges_set))
                best_path = list(e)
            
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                e = (min(best_path[i], best_path[i+1]), max(best_path[i], best_path[i+1]))
                edges_set.discard(e)
        
        return paths
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        edges = instance(*spec)
        paths = decompose_paths(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
