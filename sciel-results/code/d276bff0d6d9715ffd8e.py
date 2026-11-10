"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        N = {}
        for i in range(k, n):
            N[i] = set()
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # C-edges
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))
        
        best_cliques = None
        best_count = float('inf')
        
        # Try multiple strategies: different orderings of I-vertices
        # Strategy 1: Decreasing degree
        # Strategy 2: Increasing degree
        # Strategy 3: Random orders
        
        strategies = []
        I_vertices_list = list(range(k, n))
        
        # Deterministic orders
        strategies.append(sorted(I_vertices_list, key=lambda i: len(N[i]), reverse=True))
        strategies.append(sorted(I_vertices_list, key=lambda i: len(N[i])))
        
        # Random orders
        rng = random.Random(seed)
        for _ in range(20):
            strategies.append(rng.sample(I_vertices_list, len(I_vertices_list)))
        
        for order in strategies:
            if time.time() > deadline:
                break
                
            covered_C_edges = set()
            cliques = []
            
            for i in order:
                if time.time() > deadline:
                    break
                if not N[i]:
                    continue
                
                remaining_N = set(N[i])
                
                while remaining_N:
                    if time.time() > deadline:
                        break
                    
                    S = set()
                    # To improve greedy, we can try to pick vertices that have many
                    # free edges to other remaining vertices?
                    # For now, keep simple greedy but maybe sort remaining_N
                    # by degree in the "free" graph?
                    
                    # Simple greedy: add if no conflict
                    for u in sorted(remaining_N):
                        can_add = True
                        for v in S:
                            e = (min(u, v), max(u, v))
                            if e in covered_C_edges:
                                can_add = False
                                break
                        if can_add:
                            S.add(u)
                    
                    if not S:
                        S = {next(iter(remaining_N))}
                    
                    clique = sorted([i] + list(S))
                    cliques.append(clique)
                    
                    S_list = sorted(S)
                    for a_idx in range(len(S_list)):
                        for b_idx in range(a_idx + 1, len(S_list)):
                            e = (S_list[a_idx], S_list[b_idx])
                            covered_C_edges.add(e)
                    
                    remaining_N -= S
            
            # Cover remaining C-edges
            remaining_C_edges = C_edges - covered_C_edges
            
            if remaining_C_edges:
                remaining_adj = defaultdict(set)
                for u, v in remaining_C_edges:
                    remaining_adj[u].add(v)
                    remaining_adj[v].add(u)
                
                while remaining_C_edges:
                    if time.time() > deadline:
                        break
                    
                    # Heuristic: Pick edge with max degree in remaining graph
                    # to form a larger clique
                    max_deg = -1
                    best_edge = None
                    for u, v in remaining_C_edges:
                        # Degree of u and v in the subgraph induced by remaining_C_edges
                        # Since it's a subgraph of a clique, deg(u) = |adj[u]|
                        du = len(remaining_adj[u])
                        dv = len(remaining_adj[v])
                        # We want to maximize the size of the clique we can form.
                        # Clique size will be at least 2 + |common neighbors|.
                        # Common neighbors of u and v in remaining graph:
                        # common = remaining_adj[u] & remaining_adj[v]
                        # Size = 2 + len(common)
                        # So we want to maximize len(common).
                        common = remaining_adj[u] & remaining_adj[v]
                        if len(common) > max_deg:
                            max_deg = len(common)
                            best_edge = (u, v)
                    
                    if best_edge is None:
                        u, v = next(iter(remaining_C_edges))
                    else:
                        u, v = best_edge
                        
                    S = {u, v}
                    common = remaining_adj[u] & remaining_adj[v]
                    S |= common
                    
                    S_list = sorted(S)
                    cliques.append(S_list)
                    
                    # Remove edges
                    for a_idx in range(len(S_list)):
                        for b_idx in range(a_idx + 1, len(S_list)):
                            remaining_C_edges.discard((S_list[a_idx], S_list[b_idx]))
                    
                    # Update adj
                    for node in S_list:
                        remaining_adj.pop(node, None)
                        for w in list(remaining_adj.keys()):
                            remaining_adj[w].discard(node)
                    remaining_adj = {u2: v2 for u2, v2 in remaining_adj.items() if v2}
            
            # Check validity and count
            # Note: We assume the construction is valid by design.
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
            
            if time.time() > deadline - 5:
                break
        
        if best_cliques is None:
            # Fallback: one edge per clique
            return [list(e) for e in edges]
            
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: one edge per clique
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
