"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def greedy_clique_partition(edges, adj, n, deadline):
        """
        Greedy approach: repeatedly find the largest clique among remaining edges,
        remove all edges in that clique, repeat.
        
        To find a good clique: pick an edge, try to extend it to a maximal clique.
        Use a heuristic: prioritize edges with high common-neighbor counts.
        """
        remaining_edges = set(edges)
        cliques = []
        
        # Precompute: for each edge, how many common neighbors do its endpoints share?
        # This helps identify edges that are part of many triangles (good clique seeds)
        
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Score each remaining edge by the number of common neighbors of its endpoints
            # that are still connected to both endpoints via remaining edges
            # Approximation: use original adjacency for common neighbors, 
            # but check if those edges are still remaining
            
            best_edge = None
            best_score = -1
            best_clique = None
            
            # Sample a subset of edges if there are too many, for speed
            edge_list = list(remaining_edges)
            if len(edge_list) > 5000:
                # Sample 5000 edges
                sample = random.sample(edge_list, 5000)
            else:
                sample = edge_list
            
            for (u, v) in sample:
                # Find common neighbors of u and v
                common = adj[u] & adj[v]
                # Count how many edges from u and v to common neighbors are still remaining
                count = 0
                for w in common:
                    if (min(u, w), max(u, w)) in remaining_edges and (min(v, w), max(v, w)) in remaining_edges:
                        count += 1
                score = count
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                # No good edge found, just take any remaining edge
                best_edge = next(iter(remaining_edges))
            
            u, v = best_edge
            
            # Build a clique starting with {u, v}
            clique = {u, v}
            
            # Try to add vertices one by one
            # Candidates: common neighbors of all vertices in current clique
            while True:
                if time.time() > deadline:
                    break
                
                # Find candidates that can be added to the clique
                candidates = set(clique)
                for vert in clique:
                    candidates = candidates & adj[vert]
                candidates -= clique  # exclude vertices already in clique
                
                # Filter candidates: all edges from candidate to all clique members must be remaining
                valid_candidates = []
                for c in candidates:
                    valid = True
                    for m in clique:
                        e = (min(c, m), max(c, m))
                        if e not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        valid_candidates.append(c)
                
                if not valid_candidates:
                    break
                
                # Pick the candidate that maximizes the number of edges it shares with other candidates
                # (to allow further growth)
                best_cand = None
                best_cand_score = -1
                for c in valid_candidates:
                    # Score: number of other valid candidates that are adjacent to c
                    score = 0
                    for other in valid_candidates:
                        if other != c and other in adj[c]:
                            # Check if edge (c, other) is remaining
                            e = (min(c, other), max(c, other))
                            if e in remaining_edges:
                                score += 1
                    if score > best_cand_score:
                        best_cand_score = score
                        best_cand = c
                
                if best_cand is not None:
                    clique.add(best_cand)
                    # Remove edges from clique members to best_cand from remaining
                    for m in clique:
                        if m != best_cand:
                            e = (min(m, best_cand), max(m, best_cand))
                            remaining_edges.discard(e)
                else:
                    break
            
            # Remove all edges within the clique from remaining
            for a, b in combinations(sorted(clique), 2):
                remaining_edges.discard((a, b))
            
            # Only add clique if it has at least 2 vertices
            if len(clique) >= 2:
                cliques.append(sorted(clique))
        
        # Handle any remaining edges: each becomes its own clique of size 2
        for e in remaining_edges:
            cliques.append(sorted(e))
        
        return cliques

    def solve_instance(n, p, seed):
        edges, adj = random_graph(n, p, seed)
        return greedy_clique_partition(edges, adj, n, deadline)

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
