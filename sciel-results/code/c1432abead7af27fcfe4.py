"""Hybrid strategy: Greedy triangle removal followed by greedy set cover for remaining edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets for fast clique checking
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # 1. Find all triangles (cliques of size 3)
        # For each edge (u,v), check if there's a w connected to both.
        # To avoid duplicates, we can iterate u < v < w.
        triangles = []
        # Optimization: Iterate u, then v > u, then w > v.
        # Check if (u,v), (v,w), (u,w) are all edges.
        # Using adjacency sets:
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                # Common neighbors of u and v with index > v?
                # Actually, just checking intersection is fine for n=64.
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        triangles.append((u, v, w))
        
        # 2. Greedy Set Cover
        # State: set of uncovered edges
        uncovered_edges = set(edges)
        # Candidate cliques: Triangles first, then potentially larger ones?
        # For dense graphs, triangles cover most edges. 
        # If we just use triangles, some edges might not be in any triangle (unlikely for p=0.6, n=32/64).
        # If an edge is not in a triangle, we must use it as a size-2 clique.
        
        # Let's classify edges:
        edges_in_triangle = set()
        for t in triangles:
            u, v, w = t
            edges_in_triangle.add((min(u,v), max(u,v)))
            edges_in_triangle.add((min(v,w), max(v,w)))
            edges_in_triangle.add((min(u,w), max(u,w)))
            
        # Edges not in any triangle must be covered by size-2 cliques (themselves).
        # These are "isolated" in the sense of triangle-free.
        forced_cliques = []
        for e in uncovered_edges:
            if e not in edges_in_triangle:
                forced_cliques.append(list(e))
                uncovered_edges.remove(e)
        
        # Now we have a set of uncovered edges, all of which are in at least one triangle.
        # We want to cover these edges with triangles.
        # Greedy Set Cover:
        # 1. Map each uncovered edge to the list of triangles that cover it.
        edge_to_triangles = defaultdict(list)
        tri_index = 0
        tri_map = {}
        for t in triangles:
            tri_map[tri_index] = t
            u, v, w = t
            e1 = (min(u,v), max(u,v))
            e2 = (min(v,w), max(v,w))
            e3 = (min(u,w), max(u,w))
            if e1 in uncovered_edges: edge_to_triangles[e1].append(tri_index)
            if e2 in uncovered_edges: edge_to_triangles[e2].append(tri_index)
            if e3 in uncovered_edges: edge_to_triangles[e3].append(tri_index)
            tri_index += 1
            
        selected_cliques = list(forced_cliques)
        
        # Greedy selection
        while uncovered_edges:
            # Find the triangle that covers the most uncovered edges
            best_tri = None
            best_count = 0
            # Optimization: Only consider triangles that cover at least one uncovered edge
            # We can iterate through uncovered edges and check their triangles, 
            # or iterate through all triangles. n=64, E~1000, Triangles~10000. 
            # Iterating all triangles is 10k * 3 checks = 30k ops. Very fast.
            
            # However, as we remove edges, the count changes.
            # Let's just pick the triangle with max intersection with uncovered_edges.
            for idx, t in tri_map.items():
                u, v, w = t
                e1 = (min(u,v), max(u,v))
                e2 = (min(v,w), max(v,w))
                e3 = (min(u,w), max(u,w))
                count = 0
                if e1 in uncovered_edges: count += 1
                if e2 in uncovered_edges: count += 1
                if e3 in uncovered_edges: count += 1
                
                if count > best_count:
                    best_count = count
                    best_tri = t
            
            if best_count == 0:
                # This shouldn't happen if logic is correct, but fallback:
                # Pick any remaining edge as a size-2 clique
                e = next(iter(uncovered_edges))
                selected_cliques.append(list(e))
                uncovered_edges.remove(e)
                continue
                
            selected_cliques.append(list(best_tri))
            u, v, w = best_tri
            e1 = (min(u,v), max(u,v))
            e2 = (min(v,w), max(v,w))
            e3 = (min(u,w), max(u,w))
            uncovered_edges.discard(e1)
            uncovered_edges.discard(e2)
            uncovered_edges.discard(e3)
            
            if time.time() - start_time > time_limit:
                break

        return selected_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > time_limit:
        # If we ran out of time, return what we have for the first and empty/forced for second?
        # Better to return valid partial results if possible, but contract says "two lists".
        # If we fail, score is 0. We must ensure we finish.
        # For safety, if time is tight, return trivial solution for second?
        # Trivial solution: every edge is a clique.
        def trivial_solution(n, p, seed):
            edges = get_graph(n, p, seed)
            return [list(e) for e in edges]
        
        result2 = trivial_solution(64, 0.55, 602)
        return [result1, result2]
        
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
