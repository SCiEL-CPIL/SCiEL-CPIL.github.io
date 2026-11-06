import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def max_weight_clique(adj, weights, max_size=10):
        """Find a clique maximizing sum of weights, using greedy branch-and-bound."""
        n = len(adj)
        # Start from each vertex, try to grow clique
        best_clique = []
        best_weight = -1
        
        # Order vertices by weight descending for better pruning
        vertices = sorted(range(n), key=lambda v: -weights[v])
        
        def dfs(current_clique, candidates, current_weight):
            nonlocal best_clique, best_weight
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(current_clique)
            
            # Pruning: even if we add all remaining candidates, can we beat best?
            # Upper bound: current_weight + sum of weights of candidates (crude)
            # Better bound: current_weight + sum of top min(len(candidates), max_size-len(current_clique)) weights
            if len(current_clique) >= max_size:
                return
            
            # Simple upper bound: current_weight + sum of weights in candidates
            # This is loose but fast
            remaining_weight = sum(weights[v] for v in candidates)
            if current_weight + remaining_weight <= best_weight:
                return
            
            if not candidates:
                return
            
            # Try adding each candidate
            for i, v in enumerate(candidates):
                new_clique = current_clique + [v]
                new_weight = current_weight + weights[v]
                
                # New candidates must be neighbors of v and in candidates[i+1:]
                new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
                
                dfs(new_clique, new_candidates, new_weight)
                
                # If time is running out, stop
                if time_left() < 1.0:
                    return

        # Run DFS from a few starting points to get diversity
        # Start with the highest weight vertex
        if vertices:
            dfs([vertices[0]], [v for v in vertices if v != vertices[0] and v in adj[vertices[0]]], weights[vertices[0]])
        
        return best_clique

    def repair_partition(cliques, edges_set, adj, n):
        """Ensure every edge is covered exactly once by cliques. Split cliques if needed."""
        # Build edge-to-clique mapping
        edge_clique = {}
        valid_cliques = []
        
        for clique in cliques:
            if len(clique) < 2:
                continue
            # Check if all pairs in clique are edges
            is_clique = True
            for u, v in combinations(clique, 2):
                if v not in adj[u]:
                    is_clique = False
                    break
            if not is_clique:
                continue
            
            # Check if any edge in this clique is already covered
            has_overlap = False
            for u, v in combinations(clique, 2):
                edge = (min(u, v), max(u, v))
                if edge in edge_clique:
                    has_overlap = True
                    break
            
            if has_overlap:
                # Split: add individual edges as cliques of size 2
                for u, v in combinations(clique, 2):
                    edge = (min(u, v), max(u, v))
                    if edge not in edge_clique:
                        edge_clique[edge] = [u, v]
                        valid_cliques.append([u, v])
            else:
                for u, v in combinations(clique, 2):
                    edge = (min(u, v), max(u, v))
                    edge_clique[edge] = clique
                valid_cliques.append(clique)
        
        # Add any missing edges as individual edge cliques
        for edge in edges_set:
            if edge not in edge_clique:
                edge_clique[edge] = list(edge)
                valid_cliques.append(list(edge))
        
        # Deduplicate: if multiple cliques cover the same edge, keep only one
        # Actually, the above logic should prevent duplicates. Let's verify and clean up.
        final_cliques = []
        seen_edges = set()
        for clique in valid_cliques:
            clique_edges = set()
            for u, v in combinations(clique, 2):
                e = (min(u, v), max(u, v))
                if e in seen_edges:
                    # This clique overlaps with a previous one, so we can't use it as-is
                    # Instead, we add the uncovered edges as single edges
                    for e2 in clique_edges:
                        if e2 not in seen_edges:
                            seen_edges.add(e2)
                            final_cliques.append(list(e2))
                    break
                else:
                    clique_edges.add(e)
            else:
                # All edges in this clique are new
                seen_edges.update(clique_edges)
                final_cliques.append(clique)
        
        # Add any still-missing edges
        for edge in edges_set:
            if edge not in seen_edges:
                seen_edges.add(edge)
                final_cliques.append(list(edge))
        
        return final_cliques

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # Vertex usage count for penalty
        vertex_usage = [0] * n
        cliques = []
        
        while time_left() > 10.0:  # Keep 10s for repair and other instance
            # Weight: prefer vertices with low usage, but also consider degree
            # Weight = 1 / (1 + usage) * (degree + 1)
            weights = [1.0 / (1 + vertex_usage[v]) * (len(adj[v]) + 1) for v in range(n)]
            
            # Find max weight clique
            clique = max_weight_clique(adj, weights, max_size=8)
            
            if not clique or len(clique) < 2:
                break
            
            # Check if this clique adds any new edges
            new_edges = 0
            for u, v in combinations(clique, 2):
                e = (min(u, v), max(u, v))
                if e in edges_set:
                    new_edges += 1
            
            if new_edges == 0:
                # This clique doesn't help, skip
                continue
            
            cliques.append(clique)
            for v in clique:
                vertex_usage[v] += 1
            
            # If we've covered all edges, stop
            covered = set()
            for c in cliques:
                for u, v in combinations(c, 2):
                    covered.add((min(u, v), max(u, v)))
            
            if covered >= edges_set:
                break
        
        # Repair: ensure all edges are covered exactly once
        final_cliques = repair_partition(cliques, edges_set, adj, n)
        
        return final_cliques

    # Solve both instances
    cliques1 = solve_instance(32, 0.6, 601)
    if time_left() > 5.0:
        cliques2 = solve_instance(64, 0.55, 602)
    else:
        # Fallback: individual edges
        edges2, _ = generate_graph(64, 0.55, 602)
        cliques2 = [[u, v] for u, v in edges2]
    
    return [cliques1, cliques2]
