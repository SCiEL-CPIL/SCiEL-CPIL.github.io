import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, use 110s to be safe

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

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        if not edges:
            return []
        
        # Represent uncovered edges as a set of frozensets or just a set of tuples
        # For efficiency, we can track uncovered edges per vertex or just a global set
        # Since n is small (32, 64), a global set of tuples is fine.
        uncovered = set(edges)
        partition = []
        
        while uncovered:
            if time.time() > deadline - 5:
                # Fallback: add all remaining edges as individual cliques
                for e in uncovered:
                    partition.append(list(e))
                break
            
            # Find vertices involved in uncovered edges
            active_vertices = set()
            for u, v in uncovered:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break

            # Greedy Maximal Clique Finding on active subgraph
            # We want a clique that covers many uncovered edges.
            # Heuristic: Start with a vertex that has many uncovered edges, 
            # then greedily add vertices connected to all current clique members.
            
            # Sort active vertices by number of uncovered edges incident to them
            # This helps pick a "central" vertex to start the clique
            vert_uncovered_count = {v: 0 for v in active_vertices}
            for u, v in uncovered:
                vert_uncovered_count[u] += 1
                vert_uncovered_count[v] += 1
            
            # Start with the vertex with max uncovered count
            candidates = list(active_vertices)
            candidates.sort(key=lambda x: vert_uncovered_count[x], reverse=True)
            
            clique = []
            # Try starting from the top few candidates to find a good clique?
            # For simplicity, just start with the best one.
            # Actually, a better heuristic for "max clique" in dense graphs:
            # Iteratively pick the vertex in the candidate set with the highest degree 
            # within the candidate set.
            
            current_clique = []
            cand_set = set(active_vertices)
            
            # To improve clique size, we can be smarter.
            # Standard greedy:
            while cand_set:
                # Pick vertex in cand_set with max degree in subgraph induced by cand_set
                best_v = None
                best_deg = -1
                for v in cand_set:
                    deg = len(adj[v] & cand_set)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                
                current_clique.append(best_v)
                # Update candidates: must be neighbors of best_v
                cand_set = cand_set & adj[best_v]
            
            # current_clique is a maximal clique.
            # Check if it actually covers any uncovered edges.
            # It should, because we started from active vertices.
            # But if the clique is size 1, it covers 0 edges.
            if len(current_clique) < 2:
                # Fallback: pick an arbitrary uncovered edge
                e = next(iter(uncovered))
                current_clique = list(e)
            
            # Remove edges in current_clique from uncovered
            clique_set = set(current_clique)
            for u, v in combinations(clique_set, 2):
                edge = (min(u, v), max(u, v))
                if edge in uncovered:
                    uncovered.remove(edge)
            
            partition.append(sorted(current_clique))
            
        return partition

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
