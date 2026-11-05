# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return

            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def greedy_hitting_set(cliques, n):
        """Greedy hitting set: repeatedly pick vertex hitting most uncovered cliques."""
        if not cliques:
            return set()
        remaining = set(range(len(cliques)))
        sol = set()
        # Precompute vertex -> list of clique indices
        v_to_ci = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                v_to_ci[v].append(i)
        while remaining:
            best_v = -1
            best_cov = -1
            for v in range(n):
                if v in sol:
                    continue
                cov = sum(1 for ci in v_to_ci[v] if ci in remaining)
                if cov > best_cov:
                    best_cov = cov
                    best_v = v
            if best_v == -1 or best_cov == 0:
                break
            sol.add(best_v)
            for ci in v_to_ci[best_v]:
                remaining.discard(ci)
        return sol

    def local_search_reduce(sol, cliques, n, time_limit):
        """
        Local search: try to remove vertices that are redundant, then try swaps.
        A vertex is redundant if removing it still hits all cliques.
        """
        end_time = time.time() + time_limit
        sol = set(sol)
        
        def is_valid(s):
            for c in cliques:
                if not (c & s):
                    return False
            return True
        
        # Phase 1: Remove redundant vertices
        changed = True
        while changed and time.time() < end_time:
            changed = False
            for v in list(sol):
                if time.time() > end_time:
                    break
                trial = sol - {v}
                if is_valid(trial):
                    sol = trial
                    changed = True
        
        # Phase 2: Try 1-for-1 swaps
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for v_out in list(sol):
                if time.time() > end_time:
                    break
                for v_in in range(n):
                    if v_in in sol:
                        continue
                    trial = (sol - {v_out}) | {v_in}
                    if len(trial) < len(sol) and is_valid(trial):
                        sol = trial
                        improved = True
                        break
                    # Even same size swap might help future reductions, but skip for simplicity
        return sol

    def exact_bnb_small(cliques, n, time_limit):
        """Exact B&B for small instances (few cliques)."""
        if not cliques:
            return set()
        
        # Reduce: remove cliques that are supersets of others
        cliques_list = sorted(cliques, key=len)
        reduced = []
        for c in cliques_list:
            is_sup = False
            for rc in reduced:
                if c.issuperset(rc):
                    is_sup = True
                    break
            if not is_sup:
                reduced.append(c)
        
        if not reduced:
            return set()
        
        # If total vertices in reduced cliques is small, do exact B&B
        all_vertices = set()
        for c in reduced:
            all_vertices |= c
        
        # Greedy initial
        sol = greedy_hitting_set(reduced, n)
        best_size = len(sol)
        best_sol = set(sol)
        
        # B&B with time limit
        start = time.time()
        
        # Order cliques by size
        reduced_sorted = sorted(reduced, key=len)
        
        # Precompute vertex -> cliques it hits (in reduced)
        v_ci = defaultdict(list)
        for i, c in enumerate(reduced_sorted):
            for v in c:
                v_ci[v].append(i)
        
        def bnb(selected, remaining, depth):
            nonlocal best_size, best_sol
            if time.time() - start > time_limit:
                return
            if not remaining:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = set(selected)
                return
            if len(selected) >= best_size:
                return
            # Pick smallest remaining clique
            min_ci = min(remaining, key=lambda i: len(reduced_sorted[i]))
            min_c = reduced_sorted[min_ci]
            verts = list(min_c)
            # Order by frequency in remaining
            verts.sort(key=lambda v: sum(1 for ci in v_ci[v] if ci in remaining), reverse=True)
            for v in verts:
                if time.time() - start > time_limit:
                    return
                if v in selected:
                    continue
                new_sel = selected | {v}
                new_rem = remaining - set(v_ci[v])
                bnb(new_sel, new_rem, depth + 1)
        
        bnb(set(), set(range(len(reduced_sorted))), 0)
        return best_sol

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Strategy: 
        # 1. Get a good initial solution via greedy
        # 2. If number of cliques is small (< 200), try exact B&B with time limit
        # 3. Always apply local search reduction
        # 4. Combine: take best of exact (if fast) and greedy+local
        
        greedy_sol = greedy_hitting_set(cliques, n)
        
        # Try exact B&B if feasible
        exact_sol = None
        if len(cliques) <= 150:
            t_start = time.time()
            exact_sol = exact_bnb_small(cliques, n, min(20.0, time_budget * 0.5))
            t_elapsed = time.time() - t_start
            if exact_sol is not None and len(exact_sol) <= len(greedy_sol):
                greedy_sol = exact_sol
        
        # Local search to reduce
        remaining_time = time_budget - (time.time() - start_time)
        ls_time = max(1.0, min(remaining_time * 0.3, 30.0))
        final_sol = local_search_reduce(greedy_sol, cliques, n, ls_time)
        
        return sorted(final_sol)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 25:
            results.append(list(range(n)))
            continue
        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 15 - elapsed) / max(1, remaining_instances)
        per_instance = max(2.0, min(per_instance, deadline - 15 - elapsed))
        sol = solve_instance(n, p, seed, per_instance)
        results.append(sol)
    
    return results
# EVOLVE-BLOCK-END
