import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(graph, n):
        # graph: list of sets of neighbors
        max_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            
            # Pivot selection
            # Choose pivot u from P U X that maximizes |P ∩ N(u)|
            pivot = None
            max_count = -1
            # Try to find a good pivot
            # If P is empty, pivot doesn't matter, but X must be empty for a maximal clique
            # Standard BK: if P is empty, we are done if X is empty.
            # Actually, the base case is if P is empty and X is empty.
            # If P is empty but X is not, R is not maximal.
            
            # We need to iterate over vertices in P \ N(pivot)
            # To get P \ N(pivot), we need N(pivot).
            
            if P:
                # Pick pivot from P U X
                # Simple heuristic: first vertex in P U X
                # Or maximize intersection size
                all_candidates = list(P) + list(X)
                # Optimization: if X is large, picking from X might be better?
                # Standard is P U X.
                pivot = all_candidates[0] 
                # Better pivot:
                best_pivot = None
                best_size = -1
                for u in all_candidates:
                    count = len(P & graph[u])
                    if count > best_size:
                        best_size = count
                        best_pivot = u
                pivot = best_pivot
                
                # Iterate over P \ N(pivot)
                # We need to copy P to iterate and modify
                P_copy = list(P)
                for v in P_copy:
                    if v in graph[pivot]:
                        continue
                    
                    # Recursive call
                    R.add(v)
                    P_new = P & graph[v]
                    X_new = X & graph[v]
                    
                    bk(R, P_new, X_new)
                    
                    R.remove(v)
                    P.remove(v)
                    X.add(v)
        
        R = set()
        P = set(range(n))
        X = set()
        bk(R, P, X)
        return max_cliques

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find maximal cliques
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        # Greedy initial solution for upper bound and fallback
        # Greedy Set Cover
        uncovered = [set(c) for c in cliques]
        S = set()
        # Map vertex to list of clique indices it covers
        # This might be expensive to build fully, so do it iteratively
        remaining_cliques = [set(c) for c in cliques]
        
        # Simple Greedy:
        # At each step, pick v that covers most uncovered cliques
        # We can just loop until no uncovered cliques
        # Optimization: Pre-compute for each vertex, which cliques it belongs to?
        # Since n is small, we can just check on the fly or precompute.
        
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        covered = [False] * len(cliques)
        
        while True:
            if all(covered):
                break
            best_v = -1
            best_count = 0
            for v in range(n):
                if v in S:
                    continue
                count = 0
                for idx in vertex_to_cliques[v]:
                    if not covered[idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                break # Should not happen if valid problem
                
            S.add(best_v)
            for idx in vertex_to_cliques[best_v]:
                covered[idx] = True
        
        greedy_size = len(S)
        
        # ILP Setup
        # Variables: x_0 ... x_{n-1}
        c = [1.0] * n
        
        # Constraints: sum(x_v for v in C) >= 1 for each clique C
        # A_ub * x <= b_ub  (not used for >=)
        # LinearConstraint(A, lb, ub)
        # We want A @ x >= 1
        # So lb = 1, ub = inf
        
        A_rows = []
        for c_set in cliques:
            row = [0.0] * n
            for v in c_set:
                row[v] = 1.0
            A_rows.append(row)
        
        if not A_rows:
             return []
             
        A = A_rows
        lb = [1.0] * len(A_rows)
        ub = [float('inf')] * len(A_rows)
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = [1] * n
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.success:
                sol = res.x
                # Extract vertices
                # milp returns float, check > 0.5
                vertices = [i for i in range(n) if sol[i] > 0.5]
                # Verify validity
                is_valid = True
                for c_set in cliques:
                    if not any(v in vertices for v in c_set):
                        is_valid = False
                        break
                if is_valid:
                    return sorted(vertices)
            else:
                # Fallback to greedy
                return sorted(S)
        except Exception:
            return sorted(S)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Return best effort, maybe empty or greedy
            # We must return something valid.
            # If we ran out of time, we can return the greedy solution if we had it, 
            # but we didn't store it globally. 
            # For safety, if we run out of time, we should have already finished.
            # n=40 is small.
            pass
        res = solve_instance(n, p, seed)
        out.append(res)
        
    return out
