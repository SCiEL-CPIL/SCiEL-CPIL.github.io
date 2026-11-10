"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, edges, seed_heur, mode="max_common"):
        neighbors = [set() for _ in range(n)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            if mode == "max_common":
                # Find the best edge to start a clique: Maximize |N(u) ∩ N(v)|
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = len(common)
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)
            
            elif mode == "min_common":
                # Find the hardest edge to cover: Minimize |N(u) ∩ N(v)|
                best_edge = None
                best_score = float('inf')
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = len(common)
                    if score < best_score:
                        best_score = score
                        best_edge = (u, v)
            
            else: # "max_deg"
                # Find vertex with max degree, then pick best edge incident to it?
                # Or just max degree vertex, then expand?
                # Let's stick to edge-based selection for consistency in expansion
                # But maybe pick the edge incident to the highest degree vertex?
                # For now, let's use a hybrid: pick edge with max common, but if tie, pick from high degree.
                # Actually, let's implement "random" as a fallback or just use max_common with different shuffles.
                # Let's add a "mixed" mode: 50% max, 50% min? No, that's complex.
                # Let's just use max_common and min_common as distinct strategies.
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = len(common)
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)

            if best_edge is None:
                break
                
            u, v = best_edge
            C = [u, v]
            P = neighbors[u] & neighbors[v]
            
            # 2. Greedily expand the clique
            while P:
                # Find w in P with max degree in P
                max_deg = -1
                candidates = []
                for w in P:
                    deg = len(neighbors[w] & P)
                    if deg > max_deg:
                        max_deg = deg
                        candidates = [w]
                    elif deg == max_deg:
                        candidates.append(w)
                
                if not candidates:
                    break
                    
                # Random tie-breaking
                w = rng_heur.choice(candidates)
                C.append(w)
                P = P & neighbors[w]
            
            # 3. Remove edges of C from remaining graph
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        neighbors[a].discard(b)
                        neighbors[b].discard(a)
            
            cliques.append(C)
            
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        modes = ["max_common", "min_common"]
        
        while time.time() < time_limit:
            heur_seed = seed + attempt * 1000 + 42
            mode = modes[attempt % len(modes)]
            cliques = greedy_partition(n, edges, heur_seed, mode)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: each edge is a clique
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time: 35s for n=32, rest for n=64
    t1_limit = start_time + 35.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
