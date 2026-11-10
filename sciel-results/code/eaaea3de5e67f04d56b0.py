import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C_vertices = list(range(k))
        I_vertices = list(range(k, n))

        # For each i in I, get its neighbors in C
        # N_C(i) = adj[i] ∩ C_vertices
        N_C = [adj[i] & set(C_vertices) for i in I_vertices]

        # Strategy:
        # 1. For each i in I with non-empty N_C(i), use clique {i} ∪ N_C(i)
        #    This covers all cross edges from i and all C-C edges within N_C(i).
        # 2. Count which C-C edges are covered.
        # 3. For remaining C-C edges, cover them minimally using greedy clique cover on C.

        result_cliques = []
        covered_CC = set()

        for idx, i in enumerate(I_vertices):
            if N_C[idx]:
                clique = [i] + list(N_C[idx])
                result_cliques.append(clique)
                # Mark all C-C edges in this clique as covered
                c_part = list(N_C[idx])
                for a, b in combinations(c_part, 2):
                    covered_CC.add((min(a, b), max(a, b)))

        # Now find all C-C edges
        all_CC = set()
        for a, b in combinations(C_vertices, 2):
            all_CC.add((a, b))

        # Remaining C-C edges
        remaining = all_CC - covered_CC

        # Cover remaining C-C edges with minimum number of cliques in C
        # Since C is a clique, any subset is a valid clique.
        # We need to partition the edge set 'remaining' into minimum number of cliques.
        # This is the edge clique cover number of the graph G' = (C_vertices, remaining).
        # By the dual relationship, min clique cover of edges = max set of pairwise non-adjacent edges? No.
        # Actually, for edge clique cover: min number of cliques to cover all edges.
        # In the complement: each clique in G' corresponds to an independent set in G'^c.
        # So min edge clique cover of G' = min partition of E(G') into cliques.
        # This is equivalent to: max number of edges in a matching in G'^c? No, that's not right either.
        
        # Let's use a greedy approach: repeatedly find the largest clique in G' that covers the most uncovered edges.
        # Since G' has at most k vertices and the remaining edges are relatively few (because most C-C edges were covered by I-cliques),
        # we can afford a more careful approach.

        # Actually, let's think about this more carefully.
        # G' = (C_vertices, remaining) where remaining = E(C) \ covered_CC.
        # covered_CC consists of all pairs within each N_C(i).
        # The complement of G' within C has edges: for each pair (a,b) in C, if (a,b) is in covered_CC, it's in the complement.
        # So complement edges = union of all cliques formed by each N_C(i).
        
        # Greedy: while there are uncovered edges, find the largest clique in G' (which is a subset of C with all pairs in remaining),
        # pick the one covering the most remaining edges, remove those edges, repeat.
        
        remaining_set = set(remaining)
        
        while remaining_set:
            # Find all possible cliques in G' and pick the one covering the most edges
            # A clique in G' is a subset S of C_vertices such that all pairs in S are in remaining_set.
            # We want the S that maximizes the number of edges in S that are in remaining_set.
            # Since S is a clique in G', ALL pairs in S are in remaining_set.
            # So we just want the largest subset of C_vertices that forms a clique in G'.
            
            # Find maximum clique in G' induced by vertices that still have uncovered edges
            # Vertices with no uncovered edges can be ignored
            active_vertices = set()
            for a, b in remaining_set:
                active_vertices.add(a)
                active_vertices.add(b)
            active_vertices = list(active_vertices)
            
            if not active_vertices:
                break
            
            # Find the maximum clique in G' on active_vertices
            # For small active set, brute force all subsets
            # G' adjacency: two vertices a,b in C are adjacent in G' if (min(a,b),max(a,b)) in remaining_set
            
            # Build adjacency for G' on active vertices
            G_prime_adj = {}
            for v in active_vertices:
                G_prime_adj[v] = set()
            for a, b in remaining_set:
                if a in active_vertices and b in active_vertices:
                    G_prime_adj[a].add(b)
                    G_prime_adj[b].add(a)
            
            # Find maximum clique using simple branch and bound for small graphs
            # Since k <= 30 and we're only considering vertices with remaining edges,
            # and remaining edges should be sparse, this should be fast.
            
            # Use a simple greedy: repeatedly pick vertex with highest degree, add to clique,
            # then restrict to common neighbors.
            # Or better: try all starting vertices and expand greedily.
            
            best_clique = set()
            for start in active_vertices:
                clique = {start}
                candidates = set(G_prime_adj[start])
                # Greedily expand
                while candidates:
                    # Pick candidate with highest degree within candidates
                    best_cand = max(candidates, key=lambda x: len(candidates & G_prime_adj[x]))
                    clique.add(best_cand)
                    candidates = candidates & G_prime_adj[best_cand]
                if len(clique) > len(best_clique):
                    best_clique = clique
                    # Also compute the number of edges this clique covers
                    clique_edges = set()
                    for a, b in combinations(sorted(clique), 2):
                        if (a, b) in remaining_set or (b, a) in remaining_set:
                            clique_edges.add((min(a,b), max(a,b)))
                    
            # Compute edges covered by best_clique
            clique_edges = set()
            for a, b in combinations(sorted(best_clique), 2):
                pair = (min(a,b), max(a,b))
                if pair in remaining_set:
                    clique_edges.add(pair)
            
            if not clique_edges:
                # No edges to cover, should not happen if remaining_set is non-empty
                break
            
            # Add the clique
            result_cliques.append(sorted(best_clique))
            remaining_set -= clique_edges
        
        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
