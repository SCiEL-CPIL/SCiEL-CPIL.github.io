import time
import random
from collections import defaultdict
import numpy as np
from scipy import optimize, sparse

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, graph, n):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        if not P:
            return
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
        max_inter = -1
        pivot = pivot_candidates[0]
        for p in pivot_candidates:
            inter = len(P & graph[p])
            if inter > max_inter:
                max_inter = inter
                pivot = p
        for v in list(P - graph[pivot]):
            new_R = R | {v}
            new_P = P & graph[v]
            new_X = X & graph[v]
            yield from bron_kerbosch(new_R, new_P, new_X, graph, n)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        valid_cliques = [c for c in cliques if len(c) >= 2]
        if not valid_cliques:
            return []

        C = len(valid_cliques)

        # vertex_cliques[v] = list of clique indices containing v
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].append(i)

        # clique_vertices[i] = set of vertices in clique i
        clique_vertices = [set(c) for c in valid_cliques]

        # --- Heuristic Phase (Fallback / Initial UB) ---
        best_set = None
        best_size = float('inf')
        rng_solver = random.Random(seed + 777)

        def greedy_cover():
            hit = [False] * C
            current = set()
            while True:
                unhit = [i for i in range(C) if not hit[i]]
                if not unhit:
                    break
                best_v = -1
                best_hits = 0
                for v in range(n):
                    if v in current:
                        continue
                    hits = sum(1 for ci in vertex_cliques[v] if not hit[ci])
                    if hits > best_hits:
                        best_hits = hits
                        best_v = v
                if best_v == -1:
                    break
                current.add(best_v)
                for ci in vertex_cliques[best_v]:
                    hit[ci] = True
            return current

        def local_search_remove(current):
            improved = True
            while improved:
                improved = False
                vs = list(current)
                rng_solver.shuffle(vs)
                for v in vs:
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        c = clique_vertices[ci]
                        if not any(u in current for u in c if u != v):
                            can_remove = False
                            break
                    if can_remove:
                        current.remove(v)
                        improved = True
            return current

        # Run a few greedy restarts to get a good UB quickly
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            curr = greedy_cover()
            curr = local_search_remove(curr)
            if len(curr) < best_size:
                best_size = len(curr)
                best_set = set(curr)

        if best_set is None:
            # Fallback: all vertices in any clique
            all_vertices = set()
            for c in valid_cliques:
                all_vertices.update(c)
            best_set = all_vertices
            best_size = len(best_set)

        # --- ILP Phase (Exact / Improvement) ---
        # Try to solve the Set Cover ILP using SciPy
        # Minimize sum(x_v) s.t. sum_{v in C} x_v >= 1 for all C, x_v in {0,1}
        
        # If C is very large, building the sparse matrix might be slow, but n is small.
        # C is usually < 1000 for these parameters.
        
        try:
            # Build sparse matrix A (C x n)
            rows = []
            cols = []
            data = []
            for i, c in enumerate(valid_cliques):
                for v in c:
                    rows.append(i)
                    cols.append(v)
                    data.append(1.0)
            
            A_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(C, n))
            
            # Objective: minimize sum(x)
            c_obj = np.ones(n)
            
            # Constraints: A x >= 1
            # scipy.optimize.milp uses lb <= A x <= ub
            lb_cons = np.ones(C)
            ub_cons = np.full(C, np.inf)
            
            # Variable bounds: 0 <= x <= 1
            lb_vars = np.zeros(n)
            ub_vars = np.ones(n)
            
            # Integrality: 1 for all
            integrality = np.ones(n, dtype=int)
            
            # Time limit for ILP: leave some time for other instances
            # Total 120s. 3 instances. ~30s each? 
            # Let's give 10s per ILP call to be safe, or dynamic based on remaining time.
            remaining = deadline - time.time() - 5 # 5s safety
            if remaining < 1:
                return sorted(best_set)
            
            time_limit = max(1, min(10, remaining / 3)) # Heuristic allocation
            
            res = optimize.milp(
                c=c_obj,
                constraints=[sparse.LinAlgError if False else sparse.csr_matrix(A_sparse)], # Wait, constraints argument takes a list of LinearConstraint or sparse matrix?
                # In newer SciPy, constraints can be a list of `LinearConstraint` or a sparse matrix with lb/ub.
                # Let's use LinearConstraint to be explicit.
            )
            # Correction: 
            # constraints=[sparse.LinAlgError] is wrong.
            # Use:
            cons = sparse.LinearConstraint(A_sparse, lb=lb_cons, ub=ub_cons)
            
            res = optimize.milp(
                c=c_obj,
                constraints=cons,
                integrality=integrality,
                bounds=[(lb_vars[i], ub_vars[i]) for i in range(n)],
                options={"time_limit": time_limit}
            )
            
            if res.status == 0: # Optimal
                sol = res.x
                # sol might have small floating point errors
                selected = set()
                for i in range(n):
                    if sol[i] > 0.5:
                        selected.add(i)
                
                # Verify solution validity
                valid = True
                for i, c in enumerate(valid_cliques):
                    if not any(v in selected for v in c):
                        valid = False
                        break
                
                if valid and len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected
                    
            elif res.status == 1: # Feasible (but not necessarily optimal)
                sol = res.x
                selected = set()
                for i in range(n):
                    if sol[i] > 0.5:
                        selected.add(i)
                
                valid = True
                for i, c in enumerate(valid_cliques):
                    if not any(v in selected for v in c):
                        valid = False
                        break
                        
                if valid and len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected

        except Exception:
            pass

        return sorted(best_set)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        res = solve_instance(n, p, seed)
        results.append(res)

    return results
