import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        cliques = []
        
        # Track which K-edges are covered
        covered_kedges = set()
        
        # Step 1: For each v in I with |N(v)| >= 2, form clique {v} + N(v)
        for v in range(k, n):
            nv = neighbors[v]
            if len(nv) >= 2:
                clique = nv + [v]
                cliques.append(clique)
                # Mark K-edges within N(v) as covered
                for i in range(len(nv)):
                    for j in range(i + 1, len(nv)):
                        covered_kedges.add((min(nv[i], nv[j]), max(nv[i], nv[j])))
            elif len(nv) == 1:
                cliques.append([nv[0], v])
        
        # Step 2: Find uncovered K-edges
        uncovered = []
        for a in range(k):
            for b in range(a + 1, k):
                if (a, b) not in covered_kedges:
                    uncovered.append((a, b))
        
        # Step 3: Cover uncovered K-edges with cliques
        if uncovered:
            # Build adjacency for uncovered K-edge graph
            adj = {v: set() for v in range(k)}
            for a, b in uncovered:
                adj[a].add(b)
                adj[b].add(a)
            
            # Greedy: repeatedly find a maximal clique in the remaining graph
            remaining_edges = set(uncovered)
            rem_adj = {v: set(adj[v]) for v in range(k)}
            
            while remaining_edges:
                # Pick vertex with highest degree
                best_v = max(rem_adj.keys(), key=lambda v: len(rem_adj[v]))
                if not rem_adj[best_v]:
                    continue
                
                # Greedy clique expansion
                clique = [best_v]
                candidates = set(rem_adj[best_v])
                changed = True
                while changed:
                    changed = False
                    for c in sorted(candidates):
                        if all(c in rem_adj[cv] for cv in clique):
                            clique.append(c)
                            candidates.discard(c)
                            changed = True
                            break
                
                if len(clique) < 2:
                    # Just take an edge
                    e = next(iter(remaining_edges))
                    clique = [e[0], e[1]]
                
                cliques.append(clique)
                
                # Remove covered edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = min(clique[i], clique[j]), max(clique[i], clique[j])
                        remaining_edges.discard((a, b))
                        if a in rem_adj and b in rem_adj[a]:
                            rem_adj[a].discard(b)
                            rem_adj[b].discard(a)
        
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)

    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        neighbors = instance(n, k, p, seed)
        cliques = []
        for v in range(k, n):
            for u in neighbors[v]:
                cliques.append([u, v])
        # Cover K-edges
        for a in range(k):
            for b in range(a + 1, k):
                cliques.append([a, b])
        results.append([[int(x) for x in c] for c in cliques])

    return results
