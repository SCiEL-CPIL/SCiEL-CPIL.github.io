"""MILP-based hitting set with Bron-Kerbosch clique enumeration."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_pivot(R, P, X, adj, cliques):
        """Bron-Kerbosch with pivot to enumerate all maximal cliques."""
        if time.time() > DEADLINE:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X with max |P ∩ N(u)|
        candidates = list(P) + list(X)
        if not candidates:
            return
        
        # Pick pivot with maximum neighborhood intersection with P
        best_u = None
        best_count = -1
        for u in candidates:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_u = u
        
        # Only consider P \ N(u)
        for v in list(P - adj[best_u]):
            bron_kerbosch_pivot(R | {v}, P & adj[v], X & adj[v], adj, cliques)
            P.remove(v)
            X.add(v)

    def enumerate_maximal_cliques(n, edges):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Optimization: Isolated vertices cannot be in any clique of size >= 2.
        non_isolated = {u for u in range(n) if adj[u]}
        if not non_isolated:
            return []
            
        cliques = []
        bron_kerbosch_pivot(frozenset(), set(non_isolated), set(), adj, cliques)
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques:
            return set()
            
        uncovered_indices = set(range(len(cliques)))
        result = set()
        
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        while uncovered_indices:
            best_v = -1
            best_count = 0
            
            for v in range(n):
                if v in result:
                    continue
                count = 0
                for idx in vertex_to_cliques[v]:
                    if idx in uncovered_indices:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
                
            result.add(best_v)
            for idx in vertex_to_cliques[best_v]:
                if idx in uncovered_indices:
                    uncovered_indices.remove(idx)
        
        return result

    def local_search_minimize(hitting_set, cliques, n):
        """Try to remove vertices from the hitting set if possible."""
        hs = set(hitting_set)
        if not hs:
            return hs
            
        vertex_in_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                if v in hs:
                    vertex_in_cliques[v].append(i)

        changed = True
        while changed:
            changed = False
            for v in list(hs):
                can_remove = True
                for idx in vertex_in_cliques[v]:
                    c = cliques[idx]
                    if not any(u in hs for u in c if u != v):
                        can_remove = False
                        break
                if can_remove:
                    hs.remove(v)
                    changed = True
                    break
        return hs

    def solve_milp_hitting_set(cliques, n, time_limit=20.0):
        """Solve hitting set as MILP using scipy.optimize.milp."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
            
            m = len(cliques)
            
            # Build constraint matrix A (m x n) where A[i,v] = 1 if v in clique i
            rows = []
            cols = []
            data = []
            for i, clique in enumerate(cliques):
                for v in clique:
                    rows.append(i)
                    cols.append(v)
                    data.append(1.0)
            
            A = csr_matrix((data, (rows, cols)), shape=(m, n))
            
            # Constraints: A @ x >= 1 for each clique
            # milp uses A @ x <= b, so use -A @ x <= -1
            A_ub = -A
            
            constraint = LinearConstraint(A_ub, ub=-np.ones(m))
            bounds = Bounds(lb=0, ub=1)
            
            # Objective: minimize sum(x_v)
            c_obj = np.ones(n)
            
            # Integer variables
            integrality = np.ones(n)
            
            # Time limit
            options = {
                'time_limit': time_limit,
                'mip_rel_gap': 0.0  # Try for optimal
            }
            
            res = milp(c=c_obj, constraints=constraint, integrality=integrality, 
                       bounds=bounds, options=options)
            
            if res.status == 0 or res.status == 2:  # 0=optimal, 2=time limit with feasible
                x = res.x
                selected = set(v for v in range(n) if x[v] > 0.5)
                
                # Verify coverage
                uncovered = [c for c in cliques if not any(v in selected for v in c)]
                if uncovered:
                    # Complete with greedy
                    extra = greedy_hitting_set(uncovered, n)
                    selected.update(extra)
                
                return selected, res.status
        except Exception:
            pass
        return None, -1

    def find_min_hitting_set(cliques, n):
        """Orchestrate the finding of the minimum hitting set."""
        if not cliques:
            return []
        
        # 1. Greedy for initial bound
        sol_greedy = greedy_hitting_set(cliques, n)
        sol_greedy = local_search_minimize(sol_greedy, cliques, n)
        
        # 2. Try MILP
        elapsed = time.time() - start_time
        remaining = max(2.0, DEADLINE - elapsed - 10.0)  # Leave buffer for other instances
        
        sol_milp, status = solve_milp_hitting_set(cliques, n, time_limit=remaining)
        
        if sol_milp is not None and len(sol_milp) < len(sol_greedy):
            current_best = sol_milp
        else:
            current_best = sol_greedy
        
        # 3. Local search to polish
        current_best = local_search_minimize(current_best, cliques, n)
        
        return sorted(current_best)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all non-isolated vertices
            edges = generate_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        
        edges = generate_edges(n, p, seed)
        cliques = enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
            
        sol = find_min_hitting_set(cliques, n)
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
