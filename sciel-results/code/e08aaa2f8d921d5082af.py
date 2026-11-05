import time
import sys
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0 # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        # Use a set for O(1) lookup
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        # 1. Build adjacency matrix and list
        adj_matrix = np.zeros((n, n), dtype=bool)
        adj_list = [set() for _ in range(n)]
        for u, v in edges_set:
            adj_matrix[u, v] = True
            adj_matrix[v, u] = True
            adj_list[u].add(v)
            adj_list[v].add(u)

        # 2. Find all maximal cliques using Bron-Kerbosch with pivot
        # This is more robust than greedy peeling for finding a good candidate pool
        max_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2: # We only care about cliques of size >= 2 for edge coverage
                    max_cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P U X to minimize branching
            # Simple heuristic: pick vertex with max degree in P U X
            pivot_candidates = P.union(X)
            if not pivot_candidates:
                return
            
            # Find pivot with max connections to P
            best_pivot = None
            max_count = -1
            for u in pivot_candidates:
                count = len(P.intersection(adj_list[u]))
                if count > max_count:
                    max_count = count
                    best_pivot = u
            
            # Iterate over P \ N(u)
            # This ensures we only process vertices not connected to pivot,
            # which is sufficient to find all maximal cliques
            for v in list(P - adj_list[best_pivot]):
                bron_kerbosch(R | {v}, P & adj_list[v], X & adj_list[v])
                P.remove(v)
                X.add(v)
                
        # Initialize with all vertices that have degree > 0
        initial_P = set(range(n))
        # Filter to only vertices with at least one edge
        initial_P = {v for v in initial_P if adj_list[v]}
        
        if initial_P:
            bron_kerbosch(set(), initial_P, set())

        # 3. Formulate ILP
        # Variables: x_i for each maximal clique i
        # Objective: Minimize sum(x_i)
        # Constraints:
        # For each edge e: sum(x_i for i if e in clique_i) == 1
        
        m = len(max_cliques)
        if m == 0:
            return []

        # Build incidence matrix A where A[e_idx, c_idx] = 1 if edge e is in clique c
        edge_list = list(edges_set)
        e_to_idx = {e: i for i, e in enumerate(edge_list)}
        num_edges = len(edge_list)
        
        # We need to map edges to their index in the sorted list for consistency?
        # The set edges_set contains tuples (u,v) with u<v based on generation.
        # Let's ensure edge_list is consistent.
        
        A = np.zeros((num_edges, m), dtype=int)
        
        for c_idx, clique in enumerate(max_cliques):
            verts = list(clique)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if u > v: u, v = v, u
                    e_idx = e_to_idx[(u, v)]
                    A[e_idx, c_idx] = 1

        # 4. Solve ILP using scipy.optimize.milp
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # c: objective coefficients (all 1)
        c_obj = np.ones(m)
        
        # Constraints: A x = 1
        # LinearConstraint(A, np.ones(num_edges), np.ones(num_edges))
        
        # Bounds: x_i in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all x are integers
        integrality = np.ones(m)
        
        try:
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A, np.ones(num_edges), np.ones(num_edges)),
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0 and res.x is not None:
                # Extract solution
                selected_indices = [i for i in range(m) if res.x[i] > 0.5]
                cliques = [list(max_cliques[i]) for i in selected_indices]
                return cliques
            else:
                # Fallback if ILP fails or times out (unlikely with scipy's default, but good practice)
                # We might hit a timeout in milp if the problem is huge, but n=64 is small.
                # If it fails, we can try a greedy fallback on the maximal cliques.
                pass
        except Exception:
            pass

        # Fallback: Greedy set cover on maximal cliques
        # This ensures we return something valid if ILP fails
        uncovered = set(edges_set)
        cliques = []
        # Sort cliques by size descending to cover more edges first
        sorted_cliques = sorted(max_cliques, key=len, reverse=True)
        
        for clique in sorted_cliques:
            if not uncovered:
                break
            verts = list(clique)
            # Check if any edge in this clique is still uncovered
            has_uncovered = False
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if u > v: u, v = v, u
                    if (u, v) in uncovered:
                        has_uncovered = True
                        break
                if has_uncovered:
                    break
            
            if has_uncovered:
                cliques.append(verts)
                # Remove all edges of this clique from uncovered
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u, v = verts[i], verts[j]
                        if u > v: u, v = v, u
                        if (u, v) in uncovered:
                            uncovered.remove((u, v))
        
        # If there are still uncovered edges (shouldn't happen if maximal cliques cover all edges),
        # add them as individual edges
        for u, v in uncovered:
            cliques.append([u, v])
            
        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
