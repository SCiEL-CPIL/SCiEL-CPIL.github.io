import time
import random
import numpy as np
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj
    
    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivoting"""
        cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda u: len(adj[u] & P))
            else:
                return
            
            # Branch on vertices in P \ N(pivot)
            candidates = P - adj.get(pivot, set()) if pivot is not None else P
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                expand(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return cliques
    
    def solve_hitting_set(cliques, n, time_limit):
        """Solve minimum hitting set using branch-and-bound with LP relaxation"""
        if not cliques:
            return []
        
        # Convert cliques to list of frozensets
        clique_list = list(cliques)
        
        # Quick check: if any clique has size 1, that vertex must be in solution
        forced = set()
        remaining_cliques = []
        for c in clique_list:
            if len(c) == 1:
                forced.update(c)
            else:
                remaining_cliques.append(c)
        
        if not remaining_cliques:
            return sorted(forced)
        
        # Try LP relaxation for lower bound and initial solution
        try:
            from scipy.optimize import linprog
            import scipy.sparse as sp
            
            m = len(remaining_cliques)
            # A: m x n, A[i,j] = 1 if vertex j is in clique i
            A = np.zeros((m, n), dtype=np.float64)
            for i, c in enumerate(remaining_cliques):
                for v in c:
                    A[i, v] = 1.0
            
            c_vec = np.ones(n, dtype=np.float64)
            b = np.ones(m, dtype=np.float64)
            
            # Minimize c^T x subject to A x >= 1, 0 <= x <= 1
            # linprog minimizes c^T x subject to A_ub x <= b_ub, A_eq x = b_eq, bounds
            # So: -A x <= -1
            res = linprog(c_vec, 
                         A_ub=-A, b_ub=-b,
                         bounds=[(0, 1)] * n,
                         method='highs')
            
            if res.success and res.x is not None:
                # Round up: vertices with x > 0.5 are likely needed
                lp_sol = [i for i in range(n) if res.x[i] > 0.5]
                # Ensure it's a valid hitting set
                lp_sol_set = set(lp_sol)
                for c in remaining_cliques:
                    if not (lp_sol_set & set(c)):
                        # Add one vertex from this clique
                        for v in c:
                            if v not in lp_sol_set:
                                lp_sol_set.add(v)
                                break
                
                # Local search to improve
                best_sol = local_search_hitting_set(lp_sol_set, remaining_cliques, adj, n)
                best_sol = sorted(best_sol | forced)
                
                # Verify and return if good enough
                if is_valid_hitting_set(best_sol, clique_list):
                    return best_sol
        except Exception:
            pass
        
        # Fallback: Greedy + Local Search
        greedy_sol = greedy_hitting_set(remaining_cliques, n)
        greedy_sol = local_search_hitting_set(set(greedy_sol), remaining_cliques, adj, n)
        greedy_sol = sorted(set(greedy_sol) | forced)
        
        if is_valid_hitting_set(greedy_sol, clique_list):
            return greedy_sol
        
        # Last resort: all vertices
        return list(range(n))
    
    def is_valid_hitting_set(sol, cliques):
        sol_set = set(sol)
        for c in cliques:
            if not (sol_set & c):
                return False
        return True
    
    def greedy_hitting_set(cliques, n):
        """Greedy: pick vertex that hits the most remaining cliques"""
        if not cliques:
            return []
        
        # Count how many cliques each vertex appears in
        vertex_clique_count = [0] * n
        clique_membership = {}  # vertex -> set of clique indices
        for i, c in enumerate(cliques):
            for v in c:
                vertex_clique_count[v] += 1
                if v not in clique_membership:
                    clique_membership[v] = set()
                clique_membership[v].add(i)
        
        remaining = set(range(len(cliques)))
        sol = []
        
        while remaining:
            # Find vertex that hits most remaining cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in clique_membership:
                    hits = len(clique_membership[v] & remaining)
                    if hits > best_count:
                        best_count = hits
                        best_v = v
            
            if best_v == -1 or best_count <= 0:
                break
            
            sol.append(best_v)
            
            # Remove all cliques hit by best_v
            for i in clique_membership.get(best_v, set()):
                remaining.discard(i)
        
        return sol
    
    def local_search_hitting_set(sol_set, cliques, adj, n, max_iters=1000):
        """Local search: try to remove vertices, then try swaps"""
        sol = set(sol_set)
        
        # Phase 1: Remove redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(sol):
                # Check if removing v still hits all cliques
                sol_test = sol - {v}
                valid = True
                for c in cliques:
                    if not (sol_test & set(c)):
                        valid = False
                        break
                if valid:
                    sol = sol_test
                    changed = True
        
        # Phase 2: Try to swap - remove one, add one that's better
        improved = True
        while improved:
            improved = False
            for v in list(sol):
                # Try removing v
                sol_without = sol - {v}
                
                # Find cliques not hit by sol_without
                uncovered = []
                for c in cliques:
                    if not (sol_without & set(c)):
                        uncovered.append(c)
                
                if not uncovered:
                    # v was redundant, remove it
                    sol = sol_without
                    improved = True
                    break
                
                # Try adding one vertex to cover all uncovered cliques
                # Find candidate vertices that appear in uncovered cliques
                candidates = set()
                for c in uncovered:
                    candidates.update(c)
                candidates -= sol_without
                
                best_candidate = None
                best_size = len(sol)
                for cand in candidates:
                    # Check if adding cand covers all uncovered
                    test_sol = sol_without | {cand}
                    valid = True
                    for c in uncovered:
                        if not (test_sol & set(c)):
                            valid = False
                            break
                    if valid and len(test_sol) < best_size:
                        best_size = len(test_sol)
                        best_candidate = cand
                
                if best_candidate is not None:
                    sol = sol_without | {best_candidate}
                    improved = True
                    break
        
        return sol
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        
        if time_left < 3:
            # Not enough time, use quick greedy
            edges, adj = random_graph(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            sol = greedy_hitting_set(cliques, n)
            sol = local_search_hitting_set(set(sol), cliques, adj, n, max_iters=100)
            results.append(sorted(sol))
            continue
        
        edges, adj = random_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = bron_kerbosch(adj, n)
        
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        
        if time_left < 5:
            # Not enough time for LP, use greedy + local search
            sol = greedy_hitting_set(cliques, n)
            sol = local_search_hitting_set(set(sol), cliques, adj, n, max_iters=500)
            results.append(sorted(sol))
        else:
            # Use LP-based solver
            ilp_time = max(1, min(time_left - 5, 40))
            sol = solve_hitting_set(cliques, n, ilp_time)
            results.append(sol)
    
    return results
