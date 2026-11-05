"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_instance(n, p, seed):
        edges = set(instance(n, p, seed))
        edges = {(min(u, v), max(u, v)) for u, v in edges}
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Compute eigenvector centrality (top few iterations of power method)
        # Initialize with degree centrality
        degrees = adj.sum(axis=1).astype(float)
        centrality = degrees / (degrees.sum() + 1e-10)
        
        # A few power iterations
        for _ in range(20):
            new_cent = adj @ centrality
            norm = np.linalg.norm(new_cent)
            if norm > 0:
                new_cent /= norm
            centrality = new_cent
        
        # Sort vertices by centrality
        sorted_verts = np.argsort(centrality)[::-1].tolist()
        
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() > deadline - 5:
                # Emergency: add remaining edges as pairs
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break
            
            # Try to find best clique starting from high-centrality vertices
            best_clique = []
            best_size = 0
            
            # Try top 20 vertices by centrality
            for start_v in sorted_verts[:20]:
                if not any((min(start_v, v), max(start_v, v)) in remaining_edges for v in range(n) if v != start_v):
                    continue
                
                clique = greedy_clique(start_v, remaining_edges, n)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
                
                if best_size >= 6:  # Good enough
                    break
            
            if not best_clique:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                best_clique = [u, v]
            
            cliques.append(best_clique)
            
            # Remove edges covered by best_clique
            to_remove = set()
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    to_remove.add((min(u, v), max(u, v)))
            
            remaining_edges -= to_remove
        
        return cliques

    def greedy_clique(start_v, edges, n):
        clique = [start_v]
        # Candidates must be connected to ALL vertices in clique
        candidates = set()
        for u in range(n):
            if u != start_v:
                if (min(start_v, u), max(start_v, u)) in edges:
                    candidates.add(u)
        
        while candidates:
            # Calculate degrees within candidates subgraph
            cand_deg = {v: 0 for v in candidates}
            cand_list = list(candidates)
            for i in range(len(cand_list)):
                u = cand_list[i]
                for j in range(i + 1, len(cand_list)):
                    v = cand_list[j]
                    if (min(u, v), max(u, v)) in edges:
                        cand_deg[u] += 1
                        cand_deg[v] += 1
            
            if not any(d > 0 for d in cand_deg.values()):
                break
                
            # Pick vertex with max degree in candidate subgraph
            next_v = max(cand_deg, key=cand_deg.get)
            clique.append(next_v)
            
            # Update candidates: intersect with neighbors of next_v
            new_candidates = set()
            for u in candidates:
                if u == next_v:
                    continue
                if (min(next_v, u), max(next_v, u)) in edges:
                    new_candidates.add(u)
            candidates = new_candidates
            
        return clique

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
