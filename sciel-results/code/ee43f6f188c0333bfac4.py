"""Clique partitioning of dense random graphs using greedy maximal clique selection."""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def find_maximal_cliques(greedy_order, n, adjacency, max_cliques=500):
        """Find maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bron_kerbosch(r, p, x):
            if time.time() > deadline:
                return
            if not p and not x:
                if len(r) >= 2:
                    cliques.append(tuple(sorted(r)))
                return
            
            # Choose pivot u from p ∪ x that maximizes |N(u) ∩ p|
            u = None
            max_count = -1
            for v in list(p) + list(x):
                count = len(adjacency[v] & p)
                if count > max_count:
                    max_count = count
                    u = v
            
            # Candidates: p \ N(u)
            candidates = p - adjacency[u] if u is not None else p
            
            for v in list(candidates):
                new_r = r | {v}
                new_p = p & adjacency[v]
                new_x = x & adjacency[v]
                bron_kerbosch(new_r, new_p, new_x)
                p = p - {v}
                x = x | {v}
                if len(cliques) >= max_cliques:
                    return
        
        bron_kerbosch(set(), set(range(n)), set())
        return cliques
    
    def greedy_clique_partition(edges, n, adjacency):
        """Greedy: repeatedly pick the clique covering most uncovered edges."""
        if not edges:
            return []
        
        uncovered = set(edges)
        partition = []
        
        # Generate candidate cliques using greedy expansion
        # Start from high-degree vertices and expand
        deg = [len(adjacency[i]) for i in range(n)]
        
        # Generate candidate cliques
        candidates = []
        
        # Try expanding from each vertex
        for start in range(n):
            # Greedy clique expansion
            clique = {start}
            candidates_set = set(adjacency[start])
            
            while candidates_set:
                # Pick vertex with max connections to current clique
                best_v = None
                best_score = -1
                for v in candidates_set:
                    score = len(clique & adjacency[v])
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                if best_v is None or best_score == 0:
                    break
                
                clique.add(best_v)
                candidates_set = candidates_set & adjacency[best_v]
            
            if len(clique) >= 2:
                candidates.append(frozenset(clique))
        
        # Also try Bron-Kerbosch for more maximal cliques
        bk_cliques = find_maximal_cliques(None, n, adjacency, max_cliques=200)
        for c in bk_cliques:
            cs = frozenset(c)
            if len(cs) >= 2:
                candidates.append(cs)
        
        # Deduplicate
        candidates = list(set(candidates))
        
        # For each candidate, compute which edges it covers
        def clique_edges(clique):
            verts = list(clique)
            return set((min(a,b), max(a,b)) for a,b in combinations(verts, 2))
        
        # Precompute edge sets for candidates
        candidate_edge_sets = []
        for c in candidates:
            es = clique_edges(c)
            if es:  # Only keep candidates that cover at least one edge
                candidate_edge_sets.append((c, es))
        
        # Greedy selection
        while uncovered and time.time() < deadline:
            best_clique = None
            best_edges = None
            best_count = 0
            
            for c, es in candidate_edge_sets:
                covered = len(es & uncovered)
                if covered > best_count:
                    best_count = covered
                    best_clique = c
                    best_edges = es
            
            if best_clique is None or best_count == 0:
                # Fallback: cover remaining edges as pairs
                for e in uncovered:
                    partition.append(list(e))
                break
            
            partition.append(list(best_clique))
            uncovered -= best_edges
        
        # Handle any remaining uncovered edges
        if uncovered:
            for e in uncovered:
                partition.append(list(e))
        
        return partition
    
    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        
        # Build adjacency
        adjacency = [set() for _ in range(n)]
        for u, v in edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        partition = greedy_clique_partition(edges, n, adjacency)
        return partition
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
