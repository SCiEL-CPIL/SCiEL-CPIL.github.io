"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    TOTAL_DEADLINE = t0 + 140.0  # safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_clique_partition(n, p, seed, budget):
        edges = gen_edges(n, p, seed)
        edges_set = set(edges)
        if not edges:
            return []
        
        # Build adjacency
        G_adj = [set() for _ in range(n)]
        for u, v in edges:
            G_adj[u].add(v)
            G_adj[v].add(u)
        
        # Build conflict graph on edges
        # Two edges conflict if they share a vertex
        # We want to partition edges into cliques in the original graph
        # which means each clique in original graph corresponds to a set of edges
        # that form a complete graph on those vertices
        # Actually, a clique in the original graph with k vertices has C(k,2) edges
        # We need to cover all edges with such cliques, each edge in exactly one clique
        
        # Alternative view: find maximum cliques in G, use them to cover edges
        # Greedy: repeatedly find a large clique, remove its edges, repeat
        
        remaining = set(edges_set)
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        cliques = []
        
        # Precompute degree for priority
        def get_priority(v):
            return len(rem_adj[v])
        
        # Max clique using Bron-Kerbosch with pivot
        def max_clique(candidate_set, adj, deadline=None):
            if not candidate_set:
                return []
            
            best = []
            
            def bronk(R, P, X):
                nonlocal best
                if deadline is not None and time.time() > deadline:
                    return
                if not P and not X:
                    if len(R) > len(best):
                        best = R[:]
                    return
                # Choose pivot
                pivot = None
                max_adj_count = -1
                for v in (P | X):
                    adj_count = len(P & adj[v])
                    if adj_count > max_adj_count:
                        max_adj_count = adj_count
                        pivot = v
                if pivot is None:
                    # No pivot, iterate all of P
                    for v in list(P):
                        bronk(R + [v], P & adj[v], X & adj[v])
                        P.discard(v)
                        X.add(v)
                else:
                    for v in list(P - adj[pivot]):
                        bronk(R + [v], P & adj[v], X & adj[v])
                        P.discard(v)
                        X.add(v)
            
            bronk([], set(candidate_set), set())
            return best
        
        # Greedy clique partition
        while remaining and time.time() < budget - 2.0:
            # Find active vertices
            active = set(v for v in range(n) if rem_adj[v])
            if not active:
                break
            
            # Try to find a good clique
            # Strategy: pick vertex with highest degree, find max clique in its neighborhood + itself
            max_deg_vertex = max(active, key=get_priority)
            candidate_set = {max_deg_vertex} | (rem_adj[max_deg_vertex] & active)
            
            C = max_clique(candidate_set, rem_adj, budget - 3.0)
            
            if len(C) < 2:
                # Fallback: take one edge
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                continue
            
            # Remove all edges within C
            C_set = set(C)
            removed = 0
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining:
                        remaining.discard(e)
                        rem_adj[a].discard(b)
                        rem_adj[b].discard(a)
                        removed += 1
            cliques.append(C)
            if removed == 0:
                # Shouldn't happen, but break to avoid infinite loop
                break
        
        # Handle remaining edges
        for e in remaining:
            cliques.append([e[0], e[1]])
        
        return cliques
    
    def is_valid(partition, edges_set, G_adj):
        seen = set()
        for C in partition:
            if len(C) < 2:
                return False
            s = set(C)
            if len(s) != len(C):
                return False
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    if b not in G_adj[a]:
                        return False
                    e = (a, b) if a < b else (b, a)
                    if e in seen:
                        return False
                    seen.add(e)
        return seen == edges_set
    
    def merge_pass(cliques, G_adj):
        changed = True
        while changed:
            changed = False
            T = len(cliques)
            if T < 2:
                break
            order = sorted(range(T), key=lambda i: -len(cliques[i]))
            for idx_i in order:
                if changed:
                    break
                i = idx_i
                if i >= T:
                    continue
                Ci = set(cliques[i])
                for idx_j in order:
                    if changed:
                        break
                    j = idx_j
                    if j <= i or j >= T:
                        continue
                    Cj = set(cliques[j])
                    if not (Ci & Cj):
                        continue
                    union = Ci | Cj
                    is_clique = True
                    for u in union:
                        for v in union:
                            if u != v and v not in G_adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if not is_clique:
                        continue
                    ok2 = True
                    for k in range(T):
                        if k == i or k == j:
                            continue
                        if len(union & set(cliques[k])) > 1:
                            ok2 = False
                            break
                    if not ok2:
                        continue
                    cliques[i] = list(union)
                    del cliques[j]
                    changed = True
                    break
        return cliques
    
    result = []
    # First instance: n=32
    result.append(solve_clique_partition(32, 0.6, 601, t0 + 50.0))
    # Second instance: n=64
    if time.time() > TOTAL_DEADLINE - 2.0:
        edges = gen_edges(64, 0.55, 602)
        result.append([[u, v] for (u, v) in edges])
    else:
        result.append(solve_clique_partition(64, 0.55, 602, TOTAL_DEADLINE - 1.0))
    return result
# EVOLVE-BLOCK-END
