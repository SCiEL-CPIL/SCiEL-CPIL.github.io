"""Clique partition of dense random graphs - greedy clique extraction."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def greedy_clique_partition(n, edges, time_limit):
        if not edges:
            return []
        
        # Adjacency structure
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges and time.time() < time_limit:
            # Find a large clique in the remaining-edge-induced subgraph
            # A clique in original graph = set of vertices where all pairs have edges in remaining_edges
            
            # Greedy: pick a random edge, grow a clique around it
            best_clique_verts = None
            best_clique_edge_count = 0
            
            # Try multiple seeds
            edge_list = list(remaining_edges)
            rng = random.Random(hash(tuple(sorted(remaining_edges))) % (2**32))
            rng.shuffle(edge_list)
            
            for (su, sv) in edge_list[:30]:
                # Greedily grow a clique starting from {su, sv}
                clique = {su, sv}
                
                # Candidate vertices: those connected to all in clique
                # and having at least one remaining edge to a clique vertex
                candidates = set()
                for (a, b) in remaining_edges:
                    for vtx in (a, b):
                        if vtx not in clique:
                            candidates.add(vtx)
                
                # Iteratively add vertices
                while time.time() < time_limit:
                    # Find a vertex in candidates connected to all in clique
                    added = False
                    # Sort by degree within candidates for better growth
                    cand_list = sorted(candidates, key=lambda x: -len(adj[x] & candidates))
                    for x in cand_list:
                        if all(y in adj[x] for y in clique):
                            # Check x has remaining edges to clique
                            has_edge_to_clique = False
                            for y in clique:
                                e = (min(x, y), max(x, y))
                                if e in remaining_edges:
                                    has_edge_to_clique = True
                                    break
                            if has_edge_to_clique:
                                clique.add(x)
                                # Update candidates
                                candidates.discard(x)
                                # Remove candidates not connected to x
                                candidates = {c for c in candidates if x in adj[c]}
                                added = True
                                break
                    if not added:
                        break
                
                # Count remaining edges within this clique
                edge_count = 0
                for (a, b) in remaining_edges:
                    if a in clique and b in clique:
                        edge_count += 1
                
                if edge_count > best_clique_edge_count:
                    best_clique_edge_count = edge_count
                    best_clique_verts = set(clique)
                    if edge_count > 40:
                        break
            
            if best_clique_verts and best_clique_edge_count > 0:
                # Verify it's a valid clique (all pairs connected)
                valid = True
                verts = list(best_clique_verts)
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                        if e not in edges:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(best_clique_verts) >= 2:
                    # Remove all edges within this clique from remaining
                    for (a, b) in list(remaining_edges):
                        if a in best_clique_verts and b in best_clique_verts:
                            remaining_edges.discard((a, b))
                    cliques.append(sorted(best_clique_verts))
                else:
                    # Fallback: single edge
                    e = next(iter(remaining_edges))
                    remaining_edges.discard(e)
                    cliques.append([e[0], e[1]])
            else:
                # Fallback: single edge
                e = next(iter(remaining_edges))
                remaining_edges.discard(e)
                cliques.append([e[0], e[1]])
        
        return cliques
    
    def solve_instance(n, p, seed, time_limit):
        edges = generate_graph(n, p, seed)
        if not edges:
            return []
        return greedy_clique_partition(n, edges, time_limit)
    
    # Split time: ~65s each, with buffer
    t1_limit = start_time + 65
    t2_limit = deadline
    
    result1 = solve_instance(32, 0.6, 601, t1_limit)
    result2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
