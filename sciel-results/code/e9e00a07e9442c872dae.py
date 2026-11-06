"""Stronger solver: Bron-Kerbosch for maximal cliques, then minimum hitting set via
greedy + local search (swap/add/remove) with a time deadline."""
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # seconds; keep a safety margin before 150s kill

    def remaining_time():
        return DEADLINE - (time.time() - t0)

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_max_cliques(adj, n):
        """Return list of maximal cliques (each as a frozenset)."""
        max_cliques = []
        # Use the pivot version with precomputed adjacency sets
        adj_sets = [set(adj[i]) for i in range(n)]
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X with max |P ∩ N(u)|
            union = P | X
            if union:
                # Pick pivot with max degree into P
                best_u = max(union, key=lambda u: len(P & adj_sets[u]))
            else:
                return
            candidates = P - adj_sets[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj_sets[v], X & adj_sets[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return max_cliques

    def greedy_hit_set(cliques):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        # Map vertex -> list of clique indices it hits
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        uncovered = set(range(len(cliques)))
        chosen = set()
        
        while uncovered:
            best_v = None
            best_count = -1
            for v in vertex_to_cliques:
                if v in chosen:
                    continue
                cnt = len(vertex_to_cliques[v] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None:
                break
            chosen.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        return chosen

    def is_valid(chosen, cliques):
        for cl in cliques:
            if not cl & chosen:
                return False
        return True

    def local_search(chosen, cliques, vertex_to_cliques, n, time_limit):
        """Try to reduce |chosen| via removals, then swaps, then 2-swap."""
        chosen = set(chosen)
        start_time = time.time()
        
        # Precompute for each clique which vertices are in it (as frozenset for fast intersection)
        clique_sets = [frozenset(cl) for cl in cliques]
        
        def count_uncovered_after_removal(v, chosen_set):
            """Count how many cliques would become uncovered if we remove v from chosen_set."""
            uncovered = 0
            for cl in clique_sets:
                hits = cl & chosen_set
                if len(hits) == 1 and v in hits:
                    uncovered += 1
            return uncovered
        
        # Phase 1: try removing vertices (with precomputed critical clique tracking)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for v in list(chosen):
                if count_uncovered_after_removal(v, chosen) == 0:
                    chosen.discard(v)
                    improved = True
        
        # Phase 2: try swapping one chosen vertex for an unchosen one
        all_vertices = set()
        for cl in cliques:
            all_vertices |= cl
        
        # Precompute for each vertex which cliques it belongs to
        vertex_clique_indices = defaultdict(set)
        for i, cl in enumerate(clique_sets):
            for v in cl:
                vertex_clique_indices[v].add(i)
        
        for _ in range(15):  # more rounds of swaps
            if (time.time() - start_time) >= time_limit:
                break
            improved = False
            for v in list(chosen):
                # Find cliques that only v covers (among chosen)
                critical_cliques = set()
                for i, cl in enumerate(clique_sets):
                    hits = cl & chosen
                    if len(hits) == 1 and v in hits:
                        critical_cliques.add(i)
                
                # For each candidate replacement w not in chosen
                # w must cover all critical_cliques
                for w in all_vertices:
                    if w in chosen:
                        continue
                    # Check w covers all critical cliques
                    w_cliques = vertex_clique_indices[w]
                    if not critical_cliques.issubset(w_cliques):
                        continue
                    # Try the swap
                    new_chosen = (chosen - {v}) | {w}
                    if is_valid(new_chosen, cliques):
                        chosen = new_chosen
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Phase 3: try 2-swap (remove two, add one or two)
        for _ in range(8):
            if (time.time() - start_time) >= time_limit:
                break
            improved = False
            for v1 in list(chosen):
                for v2 in list(chosen):
                    if v1 == v2:
                        continue
                    # Try removing both v1 and v2
                    remaining = chosen - {v1, v2}
                    if is_valid(remaining, cliques):
                        chosen = remaining
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Phase 4: final removal pass
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for v in list(chosen):
                if count_uncovered_after_removal(v, chosen) == 0:
                    chosen.discard(v)
                    improved = True
        
        return chosen

    def solve_instance(n, p, seed, time_budget):
        edges = instance(n, p, seed)
        adj = build_adj(n, edges)
        
        # Find maximal cliques
        cliques = bron_kerbosch_max_cliques(adj, n)
        
        if not cliques:
            return []
        
        # Build vertex-to-clique mapping
        vertex_to_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_to_cliques[v].add(i)
        
        # Greedy initial solution
        chosen = greedy_hit_set(cliques)
        
        # Local search with remaining time
        elapsed = time.time() - t0
        ls_time = max(0.1, time_budget - elapsed)
        chosen = local_search(chosen, cliques, vertex_to_cliques, n, ls_time)
        
        return sorted(chosen)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() - t0 > DEADLINE - 3:
            # Not enough time; return what we have plus empty for rest
            results.append([])
            continue
        # Allocate time proportionally, with more for larger instances
        remaining = DEADLINE - (time.time() - t0)
        if idx == 0:
            budget = remaining * 0.3
        elif idx == 1:
            budget = remaining * 0.3
        else:
            budget = remaining * 0.4
        sol = solve_instance(n, p, seed, budget)
        results.append(sol)
    
    return results
