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
        # N_C[i] = set of C-vertices adjacent to I-vertex i
        N_C = {i: set() for i in range(k, n)}
        # N_I[j] = set of I-vertices adjacent to C-vertex j
        N_I = {j: set() for j in range(k)}
        
        for u, v in edges:
            if u < k and v >= k:
                N_C[v].add(u)
                N_I[u].add(v)
            elif v < k and u >= k:
                N_C[u].add(v)
                N_I[v].add(u)
        
        cliques = []
        
        # 1. Add star cliques for each active I-vertex
        for i in range(k, n):
            if N_C[i]:
                clique = [i] + sorted(N_C[i])
                cliques.append(clique)
        
        # 2. Identify uncovered C-edges
        # An edge (u, v) in C is covered if there exists i in I such that u in N_C[i] and v in N_C[i]
        # This is equivalent to N_I[u] intersect N_I[v] is not empty
        uncovered_C_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if not (N_I[u] & N_I[v]):
                    uncovered_C_edges.append((u, v))
        
        # 3. Cover uncovered C-edges with min cliques
        if uncovered_C_edges:
            # Greedy set cover on edges
            # Represent edges as a set
            rem_edges = set(uncovered_C_edges)
            
            while rem_edges:
                if time.time() > deadline:
                    break
                    
                # Pick an arbitrary edge
                u, v = next(iter(rem_edges))
                
                # Build a clique starting with {u, v}
                # We want to include as many other vertices w as possible
                # such that the edges (u,w), (v,w) and (w,x) for x in clique are in rem_edges?
                # No, the clique covers all edges among its vertices that are in rem_edges.
                # So we want to find a set S containing u, v that maximizes |E(rem_edges) induced by S|.
                # Greedy heuristic:
                S = {u, v}
                # Try adding vertices from the neighborhood of u or v in the remaining graph
                # Or just all vertices? k is small.
                # Let's try adding any vertex w if it adds at least one new edge to the covered set.
                
                # Current covered edges by S
                def count_covered(S):
                    cnt = 0
                    S_list = list(S)
                    for i in range(len(S_list)):
                        for j in range(i+1, len(S_list)):
                            e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                            if e in rem_edges:
                                cnt += 1
                    return cnt
                
                current_count = count_covered(S)
                
                # Try adding each vertex w in C \ S
                # To be efficient, only consider w that are incident to some edge in rem_edges
                candidates = set()
                for a, b in rem_edges:
                    candidates.add(a)
                    candidates.add(b)
                candidates -= S
                
                improved = True
                while improved:
                    improved = False
                    for w in sorted(candidates):
                        new_S = S | {w}
                        new_count = count_covered(new_S)
                        if new_count > current_count:
                            S = new_S
                            current_count = new_count
                            candidates.remove(w)
                            improved = True
                            # Re-evaluate candidates? 
                            # New edges might involve w, so other vertices might become useful
                            # But for a quick heuristic, this is fine.
                            # Actually, let's just do one pass or a few.
                            # Given the expected size is 0, this loop likely won't run.
                            pass
                    
                # Add clique S
                cliques.append(sorted(S))
                
                # Remove covered edges from rem_edges
                S_list = sorted(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        e = (S_list[i], S_list[j])
                        rem_edges.discard(e)
        
        return cliques

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
