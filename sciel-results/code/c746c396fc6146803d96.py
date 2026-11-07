import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_maximal_cliques(n, adj):
        """
        Find all maximal cliques in the graph.
        Uses Bron-Kerbosch with pivot.
        """
        cliques = []
        
        def bronk(r, p, x):
            if not p and not x:
                if len(r) >= 2: # We only care about cliques that cover edges
                    cliques.append(frozenset(r))
                return
            
            # Pivot selection
            u = None
            max_deg = -1
            # Try to find a good pivot from P U X
            candidates_pivot = list(p) + list(x)
            if candidates_pivot:
                for v in candidates_pivot:
                    deg = len(p & adj[v])
                    if deg > max_deg:
                        max_deg = deg
                        u = v
            
            if u is None:
                candidates = list(p)
            else:
                candidates = [v for v in p if v not in adj[u]]
            
            for v in candidates:
                bronk(r | {v}, p & adj[v], x & adj[v])
                p.remove(v)
                x.add(v)

        bronk(set(), set(range(n)), set())
        return cliques

    def solve_instance_mip(n, k, p, seed, time_limit=30):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all maximal cliques
        max_cliques = find_maximal_cliques(n, adj)
        
        # Filter cliques: must contain at least 2 vertices
        # Also, for efficiency, we can remove cliques that are subsets of others? 
        # No, maximal cliques are by definition not subsets of others in the maximal set.
        # However, not every edge needs to be covered by a *maximal* clique in the optimal solution?
        # Actually, yes. If an edge (u,v) is covered by a clique C, and C is not maximal, 
        # there is a maximal clique C' such that C subset C'. 
        # If we replace C with C', we cover more edges. 
        # But wait, we must cover *exactly* the edges of the graph. 
        # If C' contains edges not in C, those edges must be covered by *other* cliques in the partition.
        # If we use C' instead of C, we might double-cover some edges or leave others uncovered?
        # No, the constraint is that every edge is in *exactly one* clique.
        # If we select a maximal clique C', it covers all edges within C'.
        # If the optimal solution used a non-maximal clique C, it means the edges of C were covered by C,
        # and the edges of C' \ C were covered by other cliques.
        # If we switch to C', we now cover edges of C' \ C with C'. 
        # This might conflict if those edges were already covered by other selected cliques.
        # However, it is a known result that for the clique partition problem, 
        # there exists an optimal solution where all cliques are maximal?
        # Let's think. If C is a clique in the partition and is not maximal, let C* be a maximal clique containing C.
        # The edges in C* are covered by the union of cliques in the partition that intersect C*.
        # Since C covers all edges in C, and C* \ C has edges covered by other cliques...
        # Actually, it's safer to include ALL cliques of size >= 2 in the MIP if the number is manageable.
        # For n=90, the number of cliques can be exponential.
        # In a split graph, the structure is simpler.
        # Split graph: clique K (size k) and independent set I (size n-k).
        # Any clique in a split graph is of the form {v} U S where v in I and S subset K, OR S subset K.
        # Wait, if v in I, it can only be connected to vertices in K.
        # So a clique containing v from I must be {v} U S where S is a subset of N(v) in K.
        # Since K is a clique, any subset of K is a clique.
        # So the cliques are:
        # 1. Subsets of K (size >= 2)
        # 2. {v} U S where v in I, S subset N(v) (and S can be empty? No, size >= 2, so if S is empty, {v} is size 1. So S must be non-empty if v is in the clique? 
        #    If the clique is just {v}, size 1, invalid.
        #    If the clique is {v, u}, size 2. u must be in N(v).
        #    So for v in I, the cliques containing v are {v} U S where S is a non-empty subset of N(v).
        
        # Let's enumerate these specific cliques to keep the MIP small.
        # K = range(k)
        # I = range(k, n)
        
        candidate_cliques = []
        
        # 1. Cliques entirely in K
        # Number of subsets of K of size >= 2 is 2^k - k - 1.
        # For k=30, 2^30 is 1e9. Too many.
        # However, do we need all subsets of K?
        # If we use a subset S of K, it covers edges within S.
        # If we use a larger subset S' containing S, it covers more edges.
        # It is generally better to use larger cliques to cover more edges with one "cost" of 1.
        # Is it ever optimal to use a non-maximal clique within K?
        # Suppose K is a clique. The edges in K form a complete graph.
        # We want to partition E(K) into cliques.
        # This is equivalent to edge coloring the complete graph K_k? No, it's clique partition.
        # For a complete graph, the minimum clique partition is just the whole graph? 
        # No, the whole graph is a clique. So {K} is a partition of size 1.
        # So for the K part, we can just use K itself?
        # But K might be broken up if vertices in K are used in cliques with vertices in I?
        # If a vertex u in K is in a clique with v in I, that clique is {u, v} or {u, v, ...}.
        # The edge (u, w) for w in K \ {u} is NOT covered by {u, v, ...} unless w is also in that clique.
        # But w cannot be in a clique with v unless w is adjacent to v.
        # So if we put u in a clique with v, the edges (u, w) for other w in K must be covered by other cliques.
        
        # This structure suggests that we should focus on:
        # - The "large" clique K itself might be split.
        # - Cliques involving I vertices are "stars" from I to subsets of K.
        
        # Let's try a different MIP formulation or a smarter heuristic based on structure.
        # Given the time limit and the risk of MIP being too slow or having too many variables,
        # let's stick to a robust heuristic that exploits the split graph structure.
        
        # Strategy:
        # 1. Identify K (first k vertices) and I (rest).
        # 2. Edges in K form a complete graph.
        # 3. Edges in I are none.
        # 4. Cross edges are random.
        
        # A very strong heuristic for split graphs:
        # The clique partition of a split graph can be optimized by considering how to "pack" the independent set vertices into the clique.
        
        # Let's return to the MIP idea but restrict variables.
        # Variables x_C for each candidate clique C.
        # Objective: min sum x_C
        # Constraints: for each edge e, sum_{C contains e} x_C == 1.
        
        # To make this tractable, we only include "useful" cliques.
        # Which cliques are useful?
        # In a split graph, any maximal clique is either:
        # - A subset of K (if no I vertex is added)
        # - {v} U N(v) for some v in I (maximal w.r.t I)
        # Actually, a clique C is maximal if it cannot be extended.
        # If C subset K, it can be extended to K (unless C=K). So only K is maximal in K? 
        # No, if we don't use all of K, we can extend.
        # But in the partition, we don't have to use maximal cliques of the *graph*, we just need a set of cliques that partition the edges.
        # However, it is known that for edge clique cover/partition, we can restrict to maximal cliques?
        # Reference: "It is sufficient to consider only maximal cliques" is often true for edge cover, but for partition?
        # If C is not maximal, C subset C*. If we replace C with C*, we cover more edges.
        # The edges in C* \ C must be covered by other cliques.
        # If we replace C with C*, we "steal" coverage of edges in C* \ C from other cliques.
        # This might invalidate those other cliques if they relied on covering those edges exclusively?
        # No, the constraint is "exactly one".
        # If C* covers edge e, and another clique D also covers e, that's a conflict.
        # So if we switch C to C*, we must ensure no other selected clique covers any edge in C* \ C.
        # This makes the "maximal clique only" restriction non-trivial for partition.
        
        # Given the complexity, let's use the MIP with a restricted set of cliques:
        # 1. The full clique K.
        # 2. For each v in I, the clique {v} U N(v). (Maximal cliques involving v)
        # 3. For each v in I, and each u in N(v), the edge {u, v}. (Size 2 cliques)
        # 4. Maybe some subsets of K?
        
        # Let's try to solve with just:
        # - Maximal cliques of the graph.
        
        max_cliques = find_maximal_cliques(n, adj)
        
        # If there are too many maximal cliques, MIP might be slow.
        # For split graphs, how many maximal cliques are there?
        # Each v in I defines a unique maximal clique {v} U N(v) IF N(v) is not a subset of N(v') for another v'.
        # In random split graphs, N(v) are random subsets of K.
        # Number of maximal cliques is bounded by n.
        # Plus K itself? K is a maximal clique only if no v in I is connected to all of K?
        # If some v is connected to all K, then {v} U K is a maximal clique containing K.
        # So K might not be maximal.
        
        # Let's verify the number of maximal cliques.
        # It should be manageable (O(n) or O(n^2)).
        
        # Build MIP
        m = len(edges)
        num_cliques = len(max_cliques)
        
        if num_cliques == 0:
            # Fallback
            return [list(e) for e in edges]
            
        # Variable for each maximal clique
        # x_i = 1 if clique i is used
        c = np.ones(num_cliques)
        
        # Constraint matrix A: A[e, i] = 1 if edge e is in clique i
        A = np.zeros((m, num_cliques))
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for i, clique in enumerate(max_cliques):
            clique_list = list(clique)
            for u in clique_list:
                for v in clique_list:
                    if u < v:
                        e = (u, v)
                        if e in edge_to_idx:
                            A[edge_to_idx[e], i] = 1
                            
        # Constraints: A @ x == 1 (each edge covered exactly once)
        constraints = LinearConstraint(A, 1, 1)
        bounds = Bounds(0, 1)
        
        # Solve MIP
        try:
            res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_cliques))
            
            if res.status == 0: # Optimal
                # Extract solution
                used_cliques = []
                for i in range(num_cliques):
                    if res.x[i] > 0.5:
                        used_cliques.append(list(max_cliques[i]))
                        
                # Validate
                if used_cliques:
                    seen_edges = set()
                    valid = True
                    for c_list in used_cliques:
                        for u in c_list:
                            for v in c_list:
                                if u < v:
                                    e = (u, v)
                                    if e in seen_edges:
                                        valid = False
                                        break
                                    seen_edges.add(e)
                        if not valid:
                            break
                    if valid and seen_edges == set(edges):
                        return used_cliques
            elif res.status == 1: # Feasible
                 # Try to extract if available
                 if res.x is not None:
                     used_cliques = []
                     for i in range(num_cliques):
                         if res.x[i] > 0.5:
                             used_cliques.append(list(max_cliques[i]))
                     if used_cliques:
                         seen_edges = set()
                         valid = True
                         for c_list in used_cliques:
                             for u in c_list:
                                 for v in c_list:
                                     if u < v:
                                         e = (u, v)
                                         if e in seen_edges:
                                             valid = False
                                             break
                                         seen_edges.add(e)
                             if not valid:
                                 break
                         if valid and seen_edges == set(edges):
                             return used_cliques
                             
        except Exception:
            pass
            
        # Fallback to greedy if MIP fails or is too slow
        def greedy_cover(edges, n):
            remaining_edges = set(edges)
            cliques = []
            while remaining_edges:
                endpoints = set()
                for u, v in remaining_edges:
                    endpoints.add(u)
                    endpoints.add(v)
                
                res_adj = {v: set() for v in endpoints}
                for u, v in remaining_edges:
                    res_adj[u].add(v)
                    res_adj[v].add(u)
                
                best_c = []
                def bronk_res(r, p, x):
                    nonlocal best_c
                    if not p and not x:
                        if len(r) > len(best_c):
                            best_c = r.copy()
                        return
                    if len(r) + len(p) <= len(best_c):
                        return
                    all_pivot = list(p) + list(x)
                    if not all_pivot:
                        return
                    u = None
                    max_deg = -1
                    for v in all_pivot:
                        deg = len(p & res_adj[v])
                        if deg > max_deg:
                            max_deg = deg
                            u = v
                    if u is None:
                        candidates = list(p)
                    else:
                        candidates = [v for v in p if v not in res_adj[u]]
                    for v in candidates:
                        bronk_res(r + [v], p & res_adj[v], x & res_adj[v])
                        p.remove(v)
                        x.add(v)
                
                bronk_res([], set(endpoints), set())
                
                if not best_c:
                    u, v = next(iter(remaining_edges))
                    cliques.append([u, v])
                    remaining_edges.discard((u, v))
                    remaining_edges.discard((v, u))
                else:
                    cliques.append(best_c)
                    for i in range(len(best_c)):
                        for j in range(i + 1, len(best_c)):
                            u, v = best_c[i], best_c[j]
                            remaining_edges.discard((u, v))
                            remaining_edges.discard((v, u))
                            
            return cliques

        return greedy_cover(edges, n)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        cliques = solve_instance_mip(*spec, time_limit=20)
        
        # Validation
        valid = True
        seen_edges = set()
        orig_edges = set(instance(*spec))
        
        for c in cliques:
            if len(c) < 2:
                valid = False
                break
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (c[i], c[j])
                    if e in seen_edges:
                        valid = False
                        break
                    seen_edges.add(e)
            if not valid:
                break
        
        if not valid or seen_edges != orig_edges:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            results.append(cliques)

    return results

# Keep the required interface
def construct():
    start_time = time.time()
    deadline = 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_maximal_cliques(n, adj):
        cliques = []
        
        def bronk(r, p, x):
            if not p and not x:
                if len(r) >= 2:
                    cliques.append(frozenset(r))
                return
            
            u = None
            max_deg = -1
            candidates_pivot = list(p) + list(x)
            if candidates_pivot:
                for v in candidates_pivot:
                    deg = len(p & adj[v])
                    if deg > max_deg:
                        max_deg = deg
                        u = v
            
            if u is None:
                candidates = list(p)
            else:
                candidates = [v for v in p if v not in adj[u]]
            
            for v in candidates:
                bronk(r | {v}, p & adj[v], x & adj[v])
                p.remove(v)
                x.add(v)

        bronk(set(), set(range(n)), set())
        return cliques

    def solve_instance_mip(n, k, p, seed, time_limit=30):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        max_cliques = find_maximal_cliques(n, adj)
        
        if not max_cliques:
            return [list(e) for e in edges]
            
        m = len(edges)
        num_cliques = len(max_cliques)
        
        c = np.ones(num_cliques)
        A = np.zeros((m, num_cliques))
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for i, clique in enumerate(max_cliques):
            clique_list = list(clique)
            for u in clique_list:
                for v in clique_list:
                    if u < v:
                        e = (u, v)
                        if e in edge_to_idx:
                            A[edge_to_idx[e], i] = 1
                            
        constraints = LinearConstraint(A, 1, 1)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_cliques))
            
            if res.status in [0, 1] and res.x is not None:
                used_cliques = []
                for i in range(num_cliques):
                    if res.x[i] > 0.5:
                        used_cliques.append(list(max_cliques[i]))
                        
                if used_cliques:
                    seen_edges = set()
                    valid = True
                    for c_list in used_cliques:
                        for u in c_list:
                            for v in c_list:
                                if u < v:
                                    e = (u, v)
                                    if e in seen_edges:
                                        valid = False
                                        break
                                    seen_edges.add(e)
                        if not valid:
                            break
                    if valid and seen_edges == set(edges):
                        return used_cliques
                            
        except Exception:
            pass
            
        def greedy_cover(edges, n):
            remaining_edges = set(edges)
            cliques = []
            while remaining_edges:
                endpoints = set()
                for u, v in remaining_edges:
                    endpoints.add(u)
                    endpoints.add(v)
                
                res_adj = {v: set() for v in endpoints}
                for u, v in remaining_edges:
                    res_adj[u].add(v)
                    res_adj[v].add(u)
                
                best_c = []
                def bronk_res(r, p, x):
                    nonlocal best_c
                    if not p and not x:
                        if len(r) > len(best_c):
                            best_c = r.copy()
                        return
                    if len(r) + len(p) <= len(best_c):
                        return
                    all_pivot = list(p) + list(x)
                    if not all_pivot:
                        return
                    u = None
                    max_deg = -1
                    for v in all_pivot:
                        deg = len(p & res_adj[v])
                        if deg > max_deg:
                            max_deg = deg
                            u = v
                    if u is None:
                        candidates = list(p)
                    else:
                        candidates = [v for v in p if v not in res_adj[u]]
                    for v in candidates:
                        bronk_res(r + [v], p & res_adj[v], x & res_adj[v])
                        p.remove(v)
                        x.add(v)
                
                bronk_res([], set(endpoints), set())
                
                if not best_c:
                    u, v = next(iter(remaining_edges))
                    cliques.append([u, v])
                    remaining_edges.discard((u, v))
                    remaining_edges.discard((v, u))
                else:
                    cliques.append(best_c)
                    for i in range(len(best_c)):
                        for j in range(i + 1, len(best_c)):
                            u, v = best_c[i], best_c[j]
                            remaining_edges.discard((u, v))
                            remaining_edges.discard((v, u))
                            
            return cliques

        return greedy_cover(edges, n)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        cliques = solve_instance_mip(*spec, time_limit=20)
        
        valid = True
        seen_edges = set()
        orig_edges = set(instance(*spec))
        
        for c in cliques:
            if len(c) < 2:
                valid = False
                break
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (c[i], c[j])
                    if e in seen_edges:
                        valid = False
                        break
                    seen_edges.add(e)
            if not valid:
                break
        
        if not valid or seen_edges != orig_edges:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            results.append(cliques)

    return results
