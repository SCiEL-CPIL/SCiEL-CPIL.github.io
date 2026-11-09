import time
import random

def _generate_graph(n, p, seed):
    """Generate random graph edges using the same RNG as the seed program."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _extend_to_clique(start_vertices, adj):
    """Extend a set of vertices to a maximal clique."""
    clique = set(start_vertices)
    candidates = set()
    for v in clique:
        for u in adj[v]:
            if u not in clique:
                candidates.add(u)
    
    changed = True
    while changed:
        changed = False
        for v in list(candidates):
            if all(u in adj[v] for u in clique):
                clique.add(v)
                candidates.discard(v)
                changed = True
                for u in adj[v]:
                    if u not in clique:
                        candidates.add(u)
    
    return list(clique)

def _solve_instance(n, p, seed, deadline):
    """Solve one instance: partition edges into minimum number of cliques."""
    edges = _generate_graph(n, p, seed)
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    if not edge_set:
        return []
    
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set(edge_set)
    result = []
    
    while remaining:
        if time.time() > deadline:
            for e in remaining:
                result.append(list(e))
            break
        
        # Pick edge with most common neighbors
        best_edge = None
        best_score = -1
        for e in remaining:
            u, v = e
            score = len(adj[u] & adj[v])
            if score > best_score:
                best_score = score
                best_edge = e
        
        if best_edge is None:
            for e in remaining:
                result.append(list(e))
            break
        
        u, v = best_edge
        clique = _extend_to_clique({u, v}, adj)
        
        cl = list(clique)
        # Remove all edges in this clique from remaining
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                e = (min(cl[i], cl[j]), max(cl[i], cl[j]))
                remaining.discard(e)
        
        result.append(clique)
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 10:
            edges = _generate_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        instance_deadline = min(deadline - 5, time.time() + 65)
        result = _solve_instance(n, p, seed, instance_deadline)
        results.append(result)
    
    return results
