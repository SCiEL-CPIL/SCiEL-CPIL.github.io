import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def solve_instance(n, p, seed, time_budget):
        adj_mask_orig = get_graph(n, p, seed)
        
        if not any(adj_mask_orig):
            return []
        
        def iter_bits(x):
            while x:
                lsb = x & -x
                yield lsb.bit_length() - 1
                x ^= lsb

        def find_best_edge(adj, n):
            """
            Find the edge with the maximum number of common neighbors.
            This prioritizes edges that are part of many triangles, 
            leading to larger initial cliques.
            """
            best_u, best_v = -1, -1
            best_common = -1
            
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = (adj[u] & adj[v]).bit_count()
                        if common > best_common:
                            best_common = common
                            best_u, best_v = u, v
            
            return best_u, best_v

        def expand_clique(seed_u, seed_v, adj):
            """
            Expand an edge to a maximal clique.
            Heuristic: At each step, add the vertex from the candidate set
            that has the most neighbors within the candidate set.
            """
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            U &= ~(1 << seed_u) & ~(1 << seed_v)
            
            while U:
                candidates = list(iter_bits(U))
                if not candidates:
                    break
                
                best_u = -1
                best_score = -1
                
                for u in candidates:
                    # Score: number of neighbors in the current candidate set U
                    # This tends to lead to larger cliques
                    score = (adj[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_u = u
                
                if best_u == -1:
                    break
                
                C.append(best_u)
                U &= adj[best_u]
                U &= ~(1 << best_u)
                
            return C

        def remove_clique_edges(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def is_clique_union(C1, C2, adj):
            # Check if every vertex in C1 is connected to every vertex in C2
            for u in C1:
                for v in C2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def greedy_partition(adj_mask_orig, n, deadline):
            adj = list(adj_mask_orig)
            cliques = []
            
            while time.time() < deadline:
                u, v = find_best_edge(adj, n)
                if u == -1:
                    break
                C = expand_clique(u, v, adj)
                cliques.append(C)
                remove_clique_edges(C, adj)
            
            return cliques

        def try_merge(cliques, adj_mask_orig, n, deadline):
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                merged_indices = set()
                new_cliques = []
                i = 0
                while i < len(cliques):
                    if i in merged_indices:
                        i += 1
                        continue
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        if j in merged_indices:
                            continue
                        C_j = cliques[j]
                        if is_clique_union(C_i, C_j, adj_mask_orig):
                            new_cliques.append(C_i + C_j)
                            merged_indices.add(j)
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                    i += 1
                cliques = new_cliques
            return cliques

        def validate(cliques, adj_mask_orig, n):
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        # 1. Greedy Construction
        deadline_greedy = start_time + min(10, time_budget * 0.5)
        cliques = greedy_partition(adj_mask_orig, n, deadline_greedy)
        
        if not validate(cliques, adj_mask_orig, n):
            # Fallback: trivial partition (each edge is a clique)
            # This should rarely happen if logic is correct, but ensures validity
            cliques = []
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        cliques.append([u, v])
            return cliques

        # 2. Merging Phase
        deadline_merge = start_time + time_budget - 5
        cliques = try_merge(cliques, adj_mask_orig, n, deadline_merge)
        
        return cliques

    results = []
    # Instance 1
    t1 = min(50, TOTAL_BUDGET - 65)
    results.append(solve_instance(32, 0.6, 601, t1))
    
    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 3)
    results.append(solve_instance(64, 0.55, 602, t2))
    
    return results
