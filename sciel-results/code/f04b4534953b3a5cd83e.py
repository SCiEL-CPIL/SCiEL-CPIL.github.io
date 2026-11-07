import time
import sys
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        # C is 0..k-1
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # I is k..n-1
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_residual_clique_partition(vertices, edges):
        """
        vertices: list of vertex ids in the residual graph
        edges: list of (u, v) edges to cover
        Returns list of cliques (lists of vertices)
        """
        if not edges:
            return []
        
        # Map vertices to 0..m-1 for simplicity if needed, but we can keep original ids
        # Build adjacency
        adj = {v: set() for v in vertices}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all cliques
        # Since the graph is likely sparse/small, we can enumerate subsets?
        # No, better to find maximal cliques.
        # But for ILP, any clique is a potential column.
        # However, only maximal cliques need to be considered?
        # No, a non-maximal clique might be part of the solution.
        # But if a clique S is subset of T, and both are valid, 
        # using T is always at least as good as using S for covering edges in S, 
        # because T covers superset of edges.
        # Wait, "exactly one" constraint.
        # If we use T, it covers edges in T.
        # If we use S, it covers edges in S.
        # If we have an edge in T \ S, it must be covered by another clique.
        # It turns out that in the minimum clique partition, we can restrict to maximal cliques?
        # Not necessarily. Consider a path of 3 vertices 1-2-3.
        # Maximal cliques are {1,2} and {2,3}.
        # Partition: {1,2}, {2,3}. Cost 2.
        # If we had a triangle 1-2-3, maximal clique {1,2,3}. Cost 1.
        # So maximal cliques are good candidates.
        # Let's generate all maximal cliques.
        
        # Bron-Kerbosch for maximal cliques
        maximal_cliques = []
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(list(R))
                return
            # Pivot selection: pick u in P union X with max degree in P
            pivot = max(P.union(X), key=lambda u: len(adj[u].intersection(P))) if (P or X) else None
            candidates = P.difference(adj.get(pivot, set())) if pivot else P
            
            for v in list(candidates):
                bron_kerbosch(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)
                
        P = set(vertices)
        bron_kerbosch(set(), P, set())
        
        if not maximal_cliques:
            # Fallback: edges are cliques
            return [list(e) for e in edges]
            
        # ILP formulation
        # Variables x_i for each maximal clique
        # Constraints for each edge
        # Use scipy.optimize.milp
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            num_cliques = len(maximal_cliques)
            num_edges = len(edges)
            
            # Objective: minimize sum x_i
            c = np.ones(num_cliques)
            
            # Constraints: for each edge, sum of x_i for cliques containing it = 1
            A = np.zeros((num_edges, num_cliques))
            for i, clique in enumerate(maximal_cliques):
                clique_set = set(clique)
                for j, (u, v) in enumerate(edges):
                    if u in clique_set and v in clique_set:
                        A[j, i] = 1
            
            # Bounds: 0 <= x_i <= 1
            bounds = Bounds(lb=0, ub=1)
            
            # Integrality: all variables are binary
            integrality = np.ones(num_cliques)
            
            # Constraints: A x = 1
            constraints = LinearConstraint(A, lb=1, ub=1)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                cliques = []
                for i in range(num_cliques):
                    if res.x[i] > 0.5:
                        cliques.append(maximal_cliques[i])
                return cliques
            else:
                # Fallback to edges
                return [list(e) for e in edges]
                
        except Exception:
            # Fallback to edges
            return [list(e) for e in edges]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Return best effort or empty? Must return valid.
            # If we ran out of time, we might have partial results.
            # But we process sequentially. If we fail, we should probably return the seed-like answer for safety?
            # No, just return what we have if valid.
            pass
            
        edges = instance(n, k, p, seed)
        
        # Build adjacency for C-I and C-C
        # C: 0..k-1, I: k..n-1
        adj_C = {u: set() for u in range(k)}
        adj_I = {v: set() for v in range(k, n)}
        
        for u, v in edges:
            if u < k and v < k:
                adj_C[u].add(v)
                adj_C[v].add(u)
            elif u < k and v >= k:
                adj_C[u].add(v)
                adj_I[v].add(u)
            elif u >= k and v < k:
                adj_C[v].add(u)
                adj_I[u].add(v)
                
        cliques = []
        covered_edges = set()
        
        # Step 1: Mixed cliques
        for i in range(k, n):
            neighbors_in_C = adj_I[i]
            if neighbors_in_C:
                clique = [i] + list(neighbors_in_C)
                cliques.append(clique)
                # Mark edges covered
                # Edges (i, c) for c in neighbors
                for c in neighbors_in_C:
                    covered_edges.add((i, c))
                    covered_edges.add((c, i))
                # Edges (c1, c2) for c1, c2 in neighbors
                neigh_list = list(neighbors_in_C)
                for idx1 in range(len(neigh_list)):
                    for idx2 in range(idx1 + 1, len(neigh_list)):
                        c1 = neigh_list[idx1]
                        c2 = neigh_list[idx2]
                        covered_edges.add((c1, c2))
                        covered_edges.add((c2, c1))
                        
        # Step 2: Residual C edges
        residual_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) in adj_C[u]: # Check if edge exists in original graph
                    if (u, v) not in covered_edges:
                        residual_edges.append((u, v))
                        
        if residual_edges:
            # Solve residual
            res_cliques = solve_residual_clique_partition(list(range(k)), residual_edges)
            cliques.extend(res_cliques)
            
        results.append(cliques)
        
    return results

