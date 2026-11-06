import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 118  # Leave 2s margin for 120s limit
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def bron_kerbosch_maximal_cliques(n, adj):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot: vertex in P ∪ X with max |P ∩ N(u)|
            pivot_set = P | X
            if pivot_set:
                pivot = max(pivot_set, key=lambda u: len(P & adj[u]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P = P - {v}
                X = X | {v}
        
        expand(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques
    
    def solve_instance(n, p, seed):
        edges = get_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
        
        total_edges = len(edge_set)
        
        # Find all maximal cliques
        maximal_cliques = bron_kerbosch_maximal_cliques(n, adj)
        
        if not maximal_cliques:
            # Fallback: each edge is its own clique
            return [[u, v] for u, v in edge_set]
        
        # Convert cliques to edge sets
        clique_edges = []
        for clique in maximal_cliques:
            verts = list(clique)
            if len(verts) < 2:
                continue
            e_set = set()
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    a, b = verts[i], verts[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in edge_set:
                        e_set.add((a, b))
            if e_set:
                clique_edges.append((clique, e_set))
        
        if not clique_edges:
            return [[u, v] for u, v in edge_set]
        
        # Greedy set cover: repeatedly pick clique covering most uncovered edges
        uncovered = set(edge_set)
        selected_cliques = []
        clique_avail = list(clique_edges)
        
        while uncovered and clique_avail:
            # Find clique covering most uncovered edges
            best_idx = -1
            best_count = 0
            for i, (clique, e_set) in enumerate(clique_avail):
                count = len(e_set & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            clique, e_set = clique_avail.pop(best_idx)
            selected_cliques.append(list(clique))
            uncovered -= e_set
        
        # Fallback for remaining edges
        for u, v in uncovered:
            selected_cliques.append([u, v])
        
        # Local search: try to reduce clique count
        # Strategy: try replacing two cliques with one if possible, or reassign edges
        improved = True
        max_iters = 50
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > start_time + DEADLINE:
                break
            
            # Build edge-to-clique mapping
            edge_to_clique = {}
            for idx, clique in enumerate(selected_cliques):
                verts = clique
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        a, b = verts[i], verts[j]
                        if a > b:
                            a, b = b, a
                        edge_to_clique[(a, b)] = idx
            
            # Try merging: for each pair of cliques, check if union is a clique
            # and if we can replace them
            n_cliques = len(selected_cliques)
            merged = False
            for i in range(n_cliques):
                if merged:
                    break
                for j in range(i + 1, n_cliques):
                    if time.time() > start_time + DEADLINE:
                        return selected_cliques
                    # Check if union of vertices i and j forms a clique
                    union_verts = set(selected_cliques[i]) | set(selected_cliques[j])
                    is_clique = True
                    verts = list(union_verts)
                    for k in range(len(verts)):
                        for l in range(k + 1, len(verts)):
                            a, b = verts[k], verts[l]
                            if b not in adj[a]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if not is_clique:
                        continue
                    
                    # Check if all edges in union are currently covered by cliques i and j
                    # and no other clique covers any of these edges
                    union_edges = set()
                    for k in range(len(verts)):
                        for l in range(k + 1, len(verts)):
                            a, b = verts[k], verts[l]
                            if a > b:
                                a, b = b, a
                            if (a, b) in edge_set:
                                union_edges.add((a, b))
                    
                    # All edges in union must be covered by i or j only
                    valid = True
                    for e in union_edges:
                        if e in edge_to_clique:
                            if edge_to_clique[e] not in (i, j):
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Also ensure no edges from i or j go outside union
                    for e in edge_to_clique:
                        if edge_to_clique[e] == i or edge_to_clique[e] == j:
                            if e not in union_edges:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Replace i and j with merged clique
                    selected_cliques[i] = verts
                    selected_cliques.pop(j)
                    improved = True
                    merged = True
                    break
        
        # Remove empty or single-vertex cliques
        final_cliques = []
        for c in selected_cliques:
            if len(c) >= 2:
                final_cliques.append(c)
        
        return final_cliques
    
    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
