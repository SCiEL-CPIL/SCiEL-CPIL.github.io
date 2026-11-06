import time
from collections import defaultdict
import random as _random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to minimize branching
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Build vertex_cliques: map vertex index to set of clique indices
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
                
        # Identify active vertices
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        
        if not active_vertices:
            return []
        
        # Preprocessing: Dominance Pruning
        # If C_u is a subset of C_v, u is dominated by v.
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        keep_vertices = set()
        for v in active_vertices_sorted:
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)
                
        candidates = list(keep_vertices)
        
        if not candidates:
            return []

        # Map candidates to indices for faster lookup if needed, but we work with vertex IDs directly in sets
        # Precompute clique_vertex_lists: for each clique, which candidates can hit it?
        # Actually, we need: for each clique, the list of candidate vertices in it.
        clique_cands = []
        for ci, c in enumerate(cliques):
            cands_in_c = [v for v in c if v in keep_vertices]
            if not cands_in_c:
                # This clique cannot be hit? That implies the solution is impossible or we pruned too much.
                # But dominance pruning only removes vertices if another vertex covers all their cliques.
                # If a clique has no candidates, it means all its vertices were dominated.
                # If v is dominated by u, and v is in clique C, then u is also in C.
                # So if all vertices in C are dominated, at least one dominator must be in C.
                # So this should not happen.
                return [] 
            clique_cands.append(cands_in_c)
        
        # Greedy initialization
        def greedy_cover():
            selected = set()
            unc = set(range(m))
            while unc:
                best_v = -1
                best_cov = 0
                for v in candidates:
                    if v in selected: continue
                    cov = len(unc & vertex_cliques[v])
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                unc -= vertex_cliques[best_v]
            return selected

        init_sol = greedy_cover()
        best_size = len(init_sol)
        best_solution = set(init_sol)
        
        # B&B
        # State: selected (set of vertices), uncovered (set of clique indices)
        # We branch on the uncovered clique with the fewest available vertices.
        
        def branch_and_bound(selected, uncovered):
            nonlocal best_size, best_solution
            
            if time.time() - start_time > DEADLINE:
                return

            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return
            
            if len(selected) >= best_size:
                return
                
            # Lower bound: ceil(len(uncovered) / max_possible_coverage)
            # max_possible_coverage is the max number of uncovered cliques a single candidate can cover.
            max_cov = 0
            for v in candidates:
                if v in selected: continue
                cov = len(uncovered & vertex_cliques[v])
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                # Cannot cover any more, but uncovered is not empty -> dead end
                return

            lb = len(selected) + math.ceil(len(uncovered) / max_cov)
            if lb >= best_size:
                return
                
            # Select branch variable: uncovered clique with minimum available candidates
            target_clique_idx = -1
            min_avail = float('inf')
            target_cands = []
            
            for ci in uncovered:
                avail = 0
                cands = []
                for v in clique_cands[ci]:
                    if v not in selected:
                        avail += 1
                        cands.append(v)
                if avail < min_avail:
                    min_avail = avail
                    target_clique_idx = ci
                    target_cands = cands
                    if min_avail == 1:
                        break
                    if min_avail == 0:
                        # Should not happen if we check max_cov > 0
                        return

            if target_clique_idx == -1:
                return
            
            # Sort candidates by coverage (descending) to find good solutions early
            target_cands.sort(key=lambda v: -len(uncovered & vertex_cliques[v]))
            
            for v in target_cands:
                new_selected = selected | {v}
                new_uncovered = uncovered - vertex_cliques[v]
                branch_and_bound(new_selected, new_uncovered)
                
        initial_uncovered = set(range(m))
        branch_and_bound(set(), initial_uncovered)
        
        if best_solution is None:
            return []
            
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
