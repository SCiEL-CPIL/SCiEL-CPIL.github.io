import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        
        # Reconstruct graph
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        # C is a clique: 0 to k-1
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        # I is independent: k to n-1
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        cliques = []
        covered_internal_edges = set()
        
        # Step 1: Create cliques for each v in I using its full neighborhood in C
        for v in range(k, n):
            N_v = neighbors[v] & set(range(k))
            if len(N_v) >= 1:
                clique = [v] + sorted(list(N_v))
                cliques.append(clique)
                N_v_list = sorted(list(N_v))
                for i in range(len(N_v_list)):
                    for j in range(i + 1, len(N_v_list)):
                        u1, u2 = N_v_list[i], N_v_list[j]
                        covered_internal_edges.add((u1, u2))
            
            if time.time() > deadline:
                break

        # Step 2: Find remaining uncovered edges in C
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_internal_edges:
                    remaining_edges.add((u, w))
        
        # Step 3: Partition remaining edges into cliques using Max-Degree Heuristic
        rem_set = set(remaining_edges)
        
        # Pre-compute degrees for speed? 
        # The graph is small enough (k <= 90), so O(k^3) is fine.
        
        while rem_set:
            if time.time() > deadline:
                break
            
            # Find vertex with max degree in G_rem
            max_deg = -1
            best_u = -1
            for u in range(k):
                deg = 0
                for w in range(k):
                    if u == w: continue
                    if (min(u, w), max(u, w)) in rem_set:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_u = u
            
            if best_u == -1 or max_deg == 0:
                # This shouldn't happen if rem_set is not empty, 
                # but just in case of isolated vertices (which don't form edges)
                break
                
            u = best_u
            
            # S = {u} U {x | (u,x) in rem_set}
            S = {u}
            for x in range(k):
                if x == u: continue
                if (min(u, x), max(u, x)) in rem_set:
                    S.add(x)
            
            # Add clique S
            if len(S) >= 2:
                cliques.append(sorted(list(S)))
            
            # Remove all edges within S from rem_set
            S_list = sorted(list(S))
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    edge = (S_list[i], S_list[j])
                    rem_set.discard(edge)

        results.append(cliques)

    return results
