import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def greedy_clique_cover(edges, vertices):
        """Cover a graph on 'vertices' with minimum-ish number of cliques (greedy)."""
        if not edges:
            return []
        edges = set(edges)
        cliques = []
        while edges:
            # Find a maximal clique greedily
            # Start with edge of highest "potential" (both endpoints high degree)
            deg = {v: 0 for v in vertices}
            for u, v in edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            # Pick edge with max combined degree
            best_edge = max(edges, key=lambda e: deg.get(e[0], 0) + deg.get(e[1], 0))
            clique = {best_edge[0], best_edge[1]}
            
            # Try to extend
            changed = True
            while changed:
                changed = False
                for x in sorted(vertices - clique, key=lambda x: deg.get(x, 0), reverse=True):
                    if all((min(x, y), max(x, y)) in edges for y in clique):
                        clique.add(x)
                        changed = True
                        break  # restart to allow more additions
            
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for u, v in combinations(clique, 2):
                    e = (min(u, v), max(u, v))
                    edges.discard(e)
            else:
                e = next(iter(edges))
                cliques.append([e[0], e[1]])
                edges.discard(e)
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # All internal K edges
        all_k_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        # For each v in I, find a maximum matching in N(v) (within K)
        # Since K is a clique, any subset of N(v) forms a clique in K
        # So we want to pack as many pairs as possible: maximum matching in a complete graph on N(v)
        # That's just floor(len(N(v)) / 2) pairs
        
        cliques = []
        covered_k_edges = set()  # K-edges that are covered by triples
        
        # Sort v by |N(v)| descending to handle large neighborhoods first
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = sorted(neighbors[v])
            if not N_v:
                continue
            
            # Greedy matching: pair up neighbors
            # Use a simple greedy: repeatedly pick two unused neighbors
            used = set()
            pairs = []
            remaining = list(N_v)
            
            # Sort by some heuristic to maximize packing
            # Since all are in K (a clique), any pair works
            # Just greedily pair
            i = 0
            while i < len(remaining) - 1:
                a = remaining[i]
                b = remaining[i + 1]
                pairs.append((a, b))
                used.add(a)
                used.add(b)
                i += 2
            
            leftover = [x for x in remaining if x not in used]
            
            # Create triple cliques for each pair
            for a, b in pairs:
                cliques.append([a, b, v])
                # Mark the K-edge (a,b) as covered
                e = (min(a, b), max(a, b))
                covered_k_edges.add(e)
            
            # Create pair cliques for leftover
            for u in leftover:
                cliques.append([u, v])
        
        # Now cover remaining K-edges (those not covered by any triple)
        remaining_k_edges = all_k_edges - covered_k_edges
        
        if remaining_k_edges:
            vertices_in_rem = set()
            for u, v in remaining_k_edges:
                vertices_in_rem.add(u)
                vertices_in_rem.add(v)
            rem_cliques = greedy_clique_cover(remaining_k_edges, vertices_in_rem)
            cliques.extend(rem_cliques)
        
        # Clean up: ensure all are Python ints in lists
        clean_cliques = [[int(x) for x in c] for c in cliques]
        return clean_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    
    # If we ran out of time, fill remaining with empty (shouldn't happen for these small inputs)
    while len(results) < 3:
        results.append([])
    
    return results
