"""Stronger solver: Bron-Kerbosch for maximal cliques, then minimum hitting set via
greedy + branch-and-bound with constraint propagation + ILP polish."""
import time
import random
import sys
from collections import defaultdict
from itertools import combinations

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # seconds; keep a safety margin before 150s kill

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
        # Use the pivot version
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X with max |P ∩ N(u)|
            union = P | X
            if union:
                # Pick pivot with max degree into P
                best_u = max(union, key=lambda u: len(P & adj[u]))
            else:
                return
            candidates = P - adj[best_u]
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
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

    def branch_and_bound(cliques, n, time_limit):
        """Branch-and-bound with constraint propagation for minimum hitting set.
        Returns the best solution found within time_limit."""
        start_time = time.time()
        num_cliques = len(cliques)
        
        if num_cliques == 0:
            return set()
        
        # Precompute for each vertex which cliques it hits
        vertex_cliques = defaultdict(set)
        for i, cl in enumerate(cliques):
            for v in cl:
                vertex_cliques[v].add(i)
        
        # Sort cliques by size (ascending) - smaller cliques are more constrained
        clique_order = sorted(range(num_cliques), key=lambda i: len(cliques[i]))
        
        # Best solution found so far
        best_sol = None
        best_size = float('inf')
        
        def remaining_time():
            return time_limit - (time.time() - start_time)
        
        # Branch and bound
        # State: (chosen_set, uncovered_cliques_set, depth)
        # We'll use a stack-based approach with pruning
        
        # Initial greedy solution to set upper bound
        greedy_sol = greedy_hit_set(cliques)
        best_sol = set(greedy_sol)
        best_size = len(greedy_sol)
        
        # For B&B, we need to be smarter. Let's use a different approach:
        # Start with all cliques uncovered, try to find a solution of size k
        # for k = 0, 1, 2, ... using backtracking with pruning.
        
        # Actually, let's use a proper branch-and-bound:
        # At each step, pick the most constrained uncovered clique and branch on its vertices.
        
        def solve_bb():
            nonlocal best_sol, best_size, start_time
            
            # uncovered: set of clique indices not yet covered
            # chosen: set of vertices chosen so far
            # We'll use recursion with memoization-like pruning
            
            def backtrack(chosen, uncovered, depth):
                nonlocal best_sol, best_size, start_time
                
                if (time.time() - start_time) >= time_limit:
                    return False  # timeout
                
                if not uncovered:
                    # Found a valid solution
                    if len(chosen) < best_size:
                        best_size = len(chosen)
                        best_sol = set(chosen)
                    return True
                
                # Pruning: if current size + lower bound >= best_size, prune
                # Lower bound: need at least ceil(len(uncovered) / max_cliques_per_vertex) more
                # But a simpler bound: each new vertex covers at most max_cover cliques
                if len(chosen) >= best_size:
                    return False
                
                # Pick the most constrained uncovered clique (smallest size)
                # This is a key heuristic
                target_clique_idx = min(uncovered, key=lambda i: len(cliques[i]))
                target_clique = cliques[target_clique_idx]
                
                # For each vertex in this clique, try adding it
                # Order vertices by how many uncovered cliques they cover (descending)
                candidates = []
                for v in target_clique:
                    if v not in chosen:
                        covered = len(vertex_cliques[v] & uncovered)
                        candidates.append((covered, v))
                candidates.sort(reverse=True)
                
                for covered_count, v in candidates:
                    if (time.time() - start_time) >= time_limit:
                        return False
                    
                    new_chosen = chosen | {v}
                    new_uncovered = uncovered - vertex_cliques[v]
                    
                    # Prune: if adding this vertex doesn't help enough
                    if len(new_chosen) >= best_size:
                        continue
                    
                    # Recursive call
                    if backtrack(new_chosen, new_uncovered, depth + 1):
                        pass  # Continue searching for better solutions
            
            backtrack(set(), set(range(num_cliques)), 0)
        
        solve_bb()
        
        return best_sol if best_sol is not None else set()

    def ilp_polish(chosen, cliques, n, time_limit):
        """Solve the hitting set exactly via ILP (scipy HiGHS)."""
        if not HAVE_SCIPY:
            return chosen
        if time.time() - time.time() > time_limit:  # placeholder, actual check below
            return chosen
        
        num_cliques = len(cliques)
        num_vertices = n
        
        # Objective: minimize sum of x_v
        c = [1.0] * num_vertices
        
        # Constraints: for each clique, sum_{v in clique} x_v >= 1
        # A is num_cliques x num_vertices
        A_rows = []
        for cl in cliques:
            row = [0.0] * num_vertices
            for v in cl:
                row[v] = 1.0
            A_rows.append(row)
        
        # LinearConstraint: A @ x >= 1 for each row
        # scipy milp uses lb, ub per constraint
        lb = [1.0] * num_cliques
        ub = [float('inf')] * num_cliques
        
        # Bounds: 0 <= x_v <= 1
        var_lb = [0.0] * num_vertices
        var_ub = [1.0] * num_vertices
        
        # Integrality: all integer
        integrality = [1] * num_vertices
        
        try:
            result = milp(
                c=c,
                constraints=LinearConstraint(A_rows, lb=lb, ub=ub),
                integrality=integrality,
                bounds=Bounds(var_lb, var_ub),
                options={"time_limit": max(0.1, time_limit), "mip_rel_gap": 0.0}
            )
            if result.status == 0:  # optimal
                sol = set()
                for v in range(num_vertices):
                    if result.x[v] > 0.5:
                        sol.add(v)
                # Verify validity
                if is_valid(sol, cliques):
                    return sol
        except Exception:
            pass
        
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
        
        # Start with branch-and-bound for a good initial solution
        elapsed = time.time() - t0
        bb_time = max(1.0, time_budget * 0.5)
        chosen = branch_and_bound(cliques, n, bb_time)
        
        # If B&B didn't find anything, fall back to greedy
        if not chosen:
            chosen = greedy_hit_set(cliques)
        
        # ILP polish with remaining time
        elapsed = time.time() - t0
        ilp_time = max(0.5, time_budget - (time.time() - t0) - 1.0)
        if ilp_time > 0.5:
            chosen = ilp_polish(chosen, cliques, n, ilp_time)
        
        return sorted(chosen)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        if time.time() - t0 > DEADLINE - 10:
            # Not enough time; return what we have plus empty for rest
            results.append([])
            continue
        # Allocate time: give more to later (larger) instances
        remaining = DEADLINE - (time.time() - t0)
        if idx == 0:
            budget = min(remaining * 0.25, 30.0)
        elif idx == 1:
            budget = min(remaining * 0.35, 40.0)
        else:
            budget = min(remaining - 5.0, 60.0)
        budget = max(budget, 5.0)
        sol = solve_instance(n, p, seed, budget)
        results.append(sol)
    
    return results
