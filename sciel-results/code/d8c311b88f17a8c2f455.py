import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def bron_kerbosch_maximal_cliques(n, adj, time_limit, max_cliques=10000):
        """
        Finds maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of cliques (lists of vertices).
        """
        cliques = []
        start = time.time()
        
        def expand(R, P, X):
            if time.time() > start + time_limit:
                return
            if len(cliques) >= max_cliques:
                return
                
            if not P and not X:
                cliques.append(R[:])
                return
            
            # Pivot selection: choose u from P U X with max degree in P
            pivot = -1
            max_deg = -1
            PX = P | X
            while PX:
                bit = PX & (-PX)
                v = bit.bit_length() - 1
                PX ^= bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            # Iterate over candidates in P \ N(pivot)
            if pivot != -1:
                candidates = P & ~adj[pivot]
            else:
                candidates = P
            
            while candidates:
                if time.time() > start + time_limit:
                    return
                if len(cliques) >= max_cliques:
                    return
                
                bit = candidates & (-candidates)
                v = bit.bit_length() - 1
                candidates ^= bit
                
                new_R = R + [v]
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                expand(new_R, new_P, new_X)
                
                P &= ~bit
                X |= bit

        expand([], (1 << n) - 1, 0)
        return cliques

    def greedy_set_cover_partition(n, adj, edges, cliques, deadline):
        """
        Greedily select cliques from the provided list to cover all edges exactly once.
        """
        # Map edge to index for fast lookup? No, use a set of tuples.
        uncovered_edges = set(edges)
        selected_cliques = []
        
        # Sort cliques by size descending to prefer larger cliques
        # But we need to check which ones are still "valid" (cover only uncovered edges)
        # Since edges are removed, a clique might cover an already covered edge.
        # We only select a clique if ALL its edges are currently uncovered.
        
        # To make this efficient, we can just iterate.
        # For n=64, edges ~ 1100. Cliques might be many.
        
        # Optimization: Pre-compute edge sets for each clique
        clique_edge_sets = []
        for c in cliques:
            es = set()
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u, v = c[i], c[j]
                    if u > v: u, v = v, u
                    es.add((u, v))
            # Only keep cliques that are valid in the original graph (they should be)
            clique_edge_sets.append(es)
            
        # Sort by size descending
        indices = list(range(len(cliques)))
        indices.sort(key=lambda i: len(clique_edge_sets[i]), reverse=True)
        
        current_time = time.time()
        
        while uncovered_edges and current_time < deadline:
            found = False
            for i in indices:
                if time.time() > deadline:
                    break
                es = clique_edge_sets[i]
                # Check if all edges in this clique are still uncovered
                if es.issubset(uncovered_edges):
                    selected_cliques.append(cliques[i])
                    uncovered_edges.difference_update(es)
                    found = True
                    break
            if not found:
                # No more cliques can be selected without overlap
                # This means we might have missed some edges or the cliques don't cover everything
                # In a maximal clique enumeration, the union of all maximal cliques covers all edges.
                # But we are removing edges. If we pick a clique, we remove its edges.
                # If we get stuck, it means no single remaining clique covers *only* uncovered edges.
                # This can happen if an edge is only covered by cliques that also contain covered edges.
                # In that case, we must break the clique or handle it differently.
                # For simplicity and validity, we will fall back to covering remaining edges individually.
                break
            
        # If there are still uncovered edges, cover them as 2-cliques (edges)
        # This ensures validity.
        if uncovered_edges:
            for u, v in uncovered_edges:
                selected_cliques.append([u, v])
                
        return selected_cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Fallback: trivial partition
        trivial_partition = [[u, v] for u, v in edges]
        
        if not edges:
            return []
            
        start = time.time()
        
        # Try to find maximal cliques
        # Time limit for clique finding: half the budget
        clique_time_limit = time_limit * 0.5
        cliques = bron_kerbosch_maximal_cliques(n, adj, clique_time_limit)
        
        if not cliques:
            return trivial_partition
            
        # Try to partition using these cliques
        remaining_time = time_limit - (time.time() - start)
        partition = greedy_set_cover_partition(n, adj, edges, cliques, start + time_limit)
        
        # Validate the partition
        # 1. Every clique is a clique in G
        # 2. Every edge is covered exactly once
        edge_count = {}
        valid = True
        for c in partition:
            if len(c) < 2:
                valid = False
                break
            # Check clique property
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u, v = c[i], c[j]
                    if not (adj[u] & (1 << v)):
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
            # Count edges
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u, v = c[i], c[j]
                    if u > v: u, v = v, u
                    key = (u, v)
                    edge_count[key] = edge_count.get(key, 0) + 1
                    
        if valid:
            # Check if all original edges are covered exactly once
            if len(edge_count) == len(edges):
                for e in edges:
                    if edge_count.get(e, 0) != 1:
                        valid = False
                        break
            else:
                valid = False
                
        if valid:
            return partition
        else:
            return trivial_partition

    # Time budget allocation
    # n=32 is small, n=64 is larger.
    t1 = 20.0
    t2 = 85.0
    
    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)

    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)

    return [result1, result2]
