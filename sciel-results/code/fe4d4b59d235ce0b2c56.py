"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    def get_graph(n, p, seed):
        import random
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

    def clique_partition(n, p, seed):
        edges, adj = get_graph(n, p, seed)
        edge_list = list(edges)
        
        # Precompute for each edge, the set of common neighbors (potential clique extensions)
        # For edge (u,v), common neighbors = adj[u] & adj[v]
        
        # Greedy approach: repeatedly find the largest clique among remaining edges
        # A clique in the original graph corresponds to a set of edges that form a complete subgraph
        
        remaining_edges = set(edge_list)
        cliques = []
        
        # For efficiency, we can precompute the intersection of adjacency sets
        # But let's use a smarter greedy:
        # At each step, find a maximal clique in the subgraph induced by remaining edges
        # Actually, we want to partition edges into cliques, so we need to find cliques
        # that cover all edges exactly once.
        
        # Better approach: Use the line graph concept
        # Two edges are adjacent in the line graph if they share a vertex
        # A clique in the original graph corresponds to a set of edges that are pairwise adjacent 
        # in the line graph AND form a triangle-free structure... 
        # Actually, a set of edges forms a clique in the original graph iff:
        # - All edges share a common vertex (star) OR
        # - The edges form a triangle (3 edges) OR
        # - For larger cliques, all vertices in the clique are mutually connected
        
        # Let me reconsider. A clique of size k has C(k,2) edges. 
        # For a clique {v1, v2, ..., vk}, the edges are (vi, vj) for all i<j.
        
        # Greedy strategy: 
        # 1. Find the largest clique in the remaining graph
        # 2. Remove all edges of that clique
        # 3. Repeat until no edges remain
        # 4. For any remaining single edges, they form cliques of size 2
        
        # To find large cliques efficiently in dense graphs, we can use a heuristic:
        # Start from each edge, try to extend it to a larger clique
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
            
            # Find the best clique to add
            # Strategy: for each remaining edge, try to find the largest clique containing it
            best_clique = None
            best_size = 0
            
            # Sample edges to try (for performance)
            sample_edges = list(remaining_edges)
            if len(sample_edges) > 50:
                import random
                rng = random.Random(42)
                sample_edges = rng.sample(sample_edges, 50)
            
            for (u, v) in sample_edges:
                # Try to extend edge (u,v) to a larger clique
                # Common neighbors of u and v that are also connected to all other vertices in the clique
                common = adj[u] & adj[v]
                
                # Greedily build a clique starting from {u, v}
                clique_vertices = {u, v}
                candidates = list(common)
                
                while candidates:
                    # Find a vertex that connects to all in clique_vertices
                    found = False
                    for w in candidates[:]:
                        if all(w in adj[x] for x in clique_vertices):
                            # Check that all edges (w, x) for x in clique_vertices are in remaining_edges
                            # Actually, we need the edges to be in remaining_edges for the partition
                            # But for finding the clique structure, we just need the graph structure
                            # The constraint is that all edges of the clique must be in remaining_edges
                            edges_in_clique = set()
                            for x in clique_vertices:
                                edges_in_clique.add((min(w, x), max(w, x)))
                            if all(e in remaining_edges for e in edges_in_clique):
                                clique_vertices.add(w)
                                candidates.remove(w)
                                found = True
                                break
                    
                    if not found:
                        break
                
                if len(clique_vertices) > best_size:
                    best_size = len(clique_vertices)
                    best_clique = clique_vertices
            
            # If we found a clique of size >= 2, use it
            if best_clique is not None and len(best_clique) >= 2:
                # Add all edges of this clique to the partition
                clique_edges = set()
                verts = list(best_clique)
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                        if e in remaining_edges:
                            clique_edges.add(e)
                
                if clique_edges:
                    cliques.append(list(best_clique))
                    remaining_edges -= clique_edges
            else:
                # No clique found, take a single edge
                e = next(iter(remaining_edges))
                cliques.append([e[0], e[1]])
                remaining_edges.remove(e)
        
        # Handle any remaining edges (shouldn't happen if we didn't break early)
        if time.time() > deadline - 5.0 and remaining_edges:
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = clique_partition(n, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
