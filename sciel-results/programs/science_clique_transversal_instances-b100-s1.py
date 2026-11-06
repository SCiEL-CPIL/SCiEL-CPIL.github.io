"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_graph(n, p, seed):
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

    def bron_kerbosch(R, P, X, adj, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection to optimize
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
        
        best_pivot = None
        max_count = -1
        for v in pivot_candidates:
            count = len(P & adj[v])
            if count > max_count:
                max_count = count
                best_pivot = v
        
        # Iterate over P \ N(best_pivot)
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = P
            
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X, adj, cliques)
            P.remove(v)
            X.add(v)

    def solve_transversal(n, adj, cliques):
        if not cliques:
            return []
        
        # Filter cliques of size >= 2 (already done in bron_kerbosch)
        # If there are no edges, no cliques of size >= 2, return []
        
        # Try to use scipy.optimize.milp
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            num_vars = n
            c = np.ones(num_vars)
            
            # Constraints: sum(x_v for v in C) >= 1
            # A x >= 1
            # A is (num_cliques x n)
            A = np.zeros((len(cliques), num_vars))
            for i, C in enumerate(cliques):
                for v in C:
                    A[i, v] = 1.0
            
            # LinearConstraint(A, lb=np.ones(len(cliques)), ub=np.inf)
            constraints = [LinearConstraint(A, lb=np.ones(len(cliques)), ub=np.inf)]
            integrality = np.ones(num_vars)
            bounds = Bounds(lb=0, ub=1)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                solution = [int(x) for x in res.x]
                return [i for i, val in enumerate(solution) if val == 1]
            else:
                # Fallback if milp fails
                pass
        except Exception:
            pass
        
        # Fallback: Greedy Set Cover + Local Search
        # Greedy: pick vertex that covers most uncovered cliques
        covered = [False] * len(cliques)
        selected = []
        remaining_cliques = set(range(len(cliques)))
        
        # Precompute vertex -> list of cliques it covers
        vertex_cliques = {v: [] for v in range(n)}
        for i, C in enumerate(cliques):
            for v in C:
                vertex_cliques[v].append(i)
        
        while remaining_cliques:
            best_v = None
            max_covered = -1
            for v in range(n):
                if v in selected:
                    continue
                count = 0
                for ci in vertex_cliques[v]:
                    if ci in remaining_cliques:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_v = v
            
            if best_v is None or max_covered == 0:
                break # Should not happen if cliques are valid
                
            selected.append(best_v)
            for ci in vertex_cliques[best_v]:
                if ci in remaining_cliques:
                    remaining_cliques.remove(ci)
                    
        # Local Search: Try to remove a vertex if all its cliques are covered by others
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                # Check if removing v leaves any clique uncovered
                safe = True
                # Find cliques containing v
                for ci in vertex_cliques[v]:
                    C = cliques[ci]
                    # Check if any other vertex in selected is in C
                    has_other = False
                    for u in C:
                        if u != v and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        safe = False
                        break
                if safe:
                    selected.remove(v)
                    changed = True
                    
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Emergency fallback: return all vertices (valid but large)
            edges, adj = get_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        edges, adj = get_graph(n, p, seed)
        cliques = []
        bron_kerbosch(set(), set(range(n)), set(), adj, cliques)
        
        # If no cliques found (e.g., no edges), result is []
        if not cliques:
            results.append([])
            continue
            
        sol = solve_transversal(n, adj, cliques)
        results.append(sorted(sol))
        
    return results
# EVOLVE-BLOCK-END
