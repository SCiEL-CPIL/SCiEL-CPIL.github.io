"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def mc_clique_partition(n, p, seed):
        adj = get_graph(n, p, seed)
        cliques = []
        
        # To speed up edge finding and common neighbor calculation, 
        # we maintain a list of active edges.
        # However, for n=64, the number of edges is ~1000-2000. 
        # Iterating all pairs O(n^2) is 4096 checks, which is fast.
        # Calculating intersection of sets is O(deg).
        
        # We will track remaining edges by checking adj[u] & adj[v].
        # To avoid checking the same edge twice, we only consider u < v.
        
        while True:
            if time.time() > deadline:
                break

            # Find the edge (u, v) with the minimum number of common neighbors.
            # If no common neighbors exist, the clique size is 2.
            # We want to cover the "most constrained" edges first.
            min_common_count = float('inf')
            best_u, best_v = -1, -1
            best_common = set()
            
            # Iterate through all potential edges
            # Optimization: If we have many vertices, we can just iterate u, v.
            for u in range(n):
                if not adj[u]:
                    continue
                for v in adj[u]:
                    if u >= v:
                        continue
                    # Calculate common neighbors
                    # common = adj[u] & adj[v]
                    # Note: We only care about vertices that can form a clique with u and v.
                    # Since we are building a clique containing u and v, any w in adj[u] & adj[v]
                    # is a candidate.
                    common = adj[u] & adj[v]
                    count = len(common)
                    
                    # Tie-breaking: If counts are equal, pick the one with smaller degree sum?
                    # Or just the first one found. 
                    # The "minimum common neighborhood" heuristic suggests picking the smallest.
                    if count < min_common_count:
                        min_common_count = count
                        best_u, best_v = u, v
                        best_common = common
                        # Early exit if we found an edge with 0 common neighbors?
                        # Actually, if count is 0, it's a minimal clique (edge). 
                        # But maybe there's an edge with even fewer? No, 0 is min.
                        # However, picking an edge with 0 common neighbors is good 
                        # because it forces a size-2 clique immediately.
                        if count == 0:
                            break
                if best_u != -1 and min_common_count == 0:
                    break
            
            if best_u == -1:
                # No edges left
                break

            # Construct the clique
            # The clique consists of best_u, best_v, and potentially some from best_common.
            # To maximize efficiency (minimize number of cliques), we should include
            # as many vertices from best_common as possible, provided they form a clique
            # with each other and with u, v.
            # Since best_common = N(u) \cap N(v), any subset of best_common that is a clique
            # in the induced subgraph of best_common can be added.
            
            # Greedy approach for the clique itself:
            # Start with {u, v}.
            # Try to add vertices from best_common that are connected to all current clique members.
            
            clique = [best_u, best_v]
            candidates = list(best_common)
            
            # Sort candidates by degree in the remaining graph? 
            # Or just iterate.
            # To make it a valid clique, we need to ensure new vertex connects to ALL in clique.
            # Since candidates are neighbors of u and v, they connect to u and v.
            # We just need them to connect to previously added candidates.
            
            # This is essentially finding a clique in the subgraph induced by best_common.
            # But best_common might be large. 
            # However, we are already in a "min common" context, so best_common is likely small.
            # If best_common is large, we might want to take a maximal clique from it.
            
            # Let's do a simple greedy:
            # Sort candidates by degree in the induced subgraph of candidates (descending).
            # This helps build a larger clique.
            
            # Calculate degrees within candidates
            cand_set = set(candidates)
            deg_in_cand = {c: len(adj[c] & cand_set) for c in candidates}
            candidates.sort(key=lambda x: deg_in_cand[x], reverse=True)
            
            for c in candidates:
                # Check if c connects to all vertices in current clique
                # We know c connects to u and v.
                # We need to check c connects to other members in clique[2:]
                is_connected = True
                for member in clique[2:]:
                    if member not in adj[c]:
                        is_connected = False
                        break
                if is_connected:
                    clique.append(c)
            
            # Remove edges covered by this clique
            for i in range(len(clique)):
                u_c = clique[i]
                for j in range(i + 1, len(clique)):
                    v_c = clique[j]
                    if v_c in adj[u_c]:
                        adj[u_c].remove(v_c)
                    if u_c in adj[v_c]:
                        adj[v_c].remove(u_c)
            
            cliques.append(clique)
            
        # Fallback for time limit: cover remaining edges as size-2 cliques
        remaining_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining_edges.add((u, v))
        
        for u, v in remaining_edges:
            cliques.append([u, v])
            
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = mc_clique_partition(n, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
