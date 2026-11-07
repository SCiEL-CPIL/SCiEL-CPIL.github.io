"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph_ilp(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        edge_set = set(edges)
        num_edges = len(edge_set)
        
        # Build adjacency for clique checking
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Generate Candidate Cliques
        # In a split graph, maximal cliques are:
        # 1. C (the clique part)
        # 2. {v} U N(v) for each v in I
        # Any clique in the graph is a subset of one of these maximal cliques.
        # However, for a minimum edge clique cover, we only need to consider 
        # "maximal" cliques in the sense of inclusion? 
        # Not necessarily. A smaller clique might be part of an optimal solution 
        # if it allows better packing? 
        # Actually, if K1 is a clique and K2 is a clique, and K1 subset K2, 
        # replacing K1 with K2 in a cover might cause overlaps. 
        # But in an Exact Cover (partition), if we use a larger clique, we cover more edges.
        # The problem is equivalent to finding a minimum number of cliques such that 
        # every edge is in exactly one.
        
        # Let's generate all cliques of size >= 2?
        # For n=90, k=30.
        # Maximal cliques are manageable.
        # But subsets of maximal cliques are also candidates.
        # If we restrict to maximal cliques, is it sufficient?
        # Consider two edges (u,v) and (u,w) where {u,v,w} is a clique.
        # If we use clique {u,v,w}, it covers both.
        # If we use {u,v} and {u,w}, they share edge (u,v)? No, they share vertex u.
        # Edge (u,v) is in {u,v}. Edge (u,w) is in {u,w}. Edge (v,w) is NOT covered.
        # So we'd need {v,w} too. Total 3 cliques.
        # Using {u,v,w} is 1 clique.
        # So larger cliques are better for covering.
        # The only reason to use a non-maximal clique is if the maximal clique contains edges 
        # that are "better" covered by other cliques? 
        # In an exact partition, edges are disjoint.
        # If we pick a maximal clique M, we commit to covering all edges in M.
        # If some edge in M is "needed" by another clique K, we have a conflict.
        # So the set of candidates should probably include subsets as well.
        
        # However, generating all subsets is exponential.
        # Let's try a heuristic generation of candidates:
        # 1. All maximal cliques.
        # 2. For each maximal clique, maybe some large subsets?
        
        # Actually, for Split Graphs, there is a known polynomial time algorithm?
        # The task mentions Erdős-Ordman-Zalcstein.
        # Let's stick to ILP with a good candidate set.
        
        candidates = []
        
        # 1. The clique C
        if k >= 2:
            candidates.append(list(range(k)))
            
        # 2. Maximal cliques {v} U N(v)
        for v in range(k, n):
            neighbors = [u for u in range(k) if u in adj[v]]
            if len(neighbors) + 1 >= 2:
                candidates.append([v] + neighbors)
                
        # 3. Add some smaller cliques?
        # To avoid missing optimal solutions that require smaller cliques to resolve conflicts,
        # we can add all cliques of size 2 and 3?
        # Size 2 is just edges.
        # Size 3: triangles.
        # In split graphs, triangles are either in C or involve one I vertex and two C vertices.
        
        # Let's add all edges as candidates (size 2 cliques).
        # This guarantees a feasible solution exists.
        for e in edge_set:
            candidates.append(list(e))
            
        # Add all triangles?
        # Triangles in C: C is a clique, so any 3 vertices in C form a triangle.
        # Triangles with I: {v, u1, u2} where v in I, u1,u2 in C, and u1,u2 in N(v).
        
        # Generating all triangles might be expensive if k is large, but k<=30.
        # Number of triangles in C: C(30,3) = 4060.
        # Number of triangles with I: For each v, C(|N(v)|, 2). |N(v)| ~ 0.7*30 = 21. C(21,2)=210.
        # Total ~ 4000 + 60*210 ~ 16600. Manageable.
        
        # Let's add all cliques of size 3.
        # 3a. Subsets of C
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j+1, k):
                    candidates.append([i, j, l])
                    
        # 3b. {v, u1, u2}
        for v in range(k, n):
            neighbors = [u for u in range(k) if u in adj[v]]
            # neighbors is a subset of C, so it's a clique.
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    u1 = neighbors[i]
                    u2 = neighbors[j]
                    candidates.append([v, u1, u2])
                    
        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for c in candidates:
            # Sort vertices for consistent hashing
            sorted_c = tuple(sorted(c))
            if sorted_c not in seen:
                seen.add(sorted_c)
                unique_candidates.append(list(sorted_c))
                
        num_candidates = len(unique_candidates)
        
        # If ILP is too big or slow, we might need to prune.
        # But let's try first.
        
        # Build incidence matrix A (num_edges x num_candidates)
        # A[e_idx, c_idx] = 1 if edge e is in clique c
        
        # Map edge to index
        edge_to_idx = {}
        for idx, e in enumerate(edge_set):
            edge_to_idx[e] = idx
            
        A = np.zeros((num_edges, num_candidates))
        
        for c_idx, c in enumerate(unique_candidates):
            # c is a list of vertices
            # Generate all pairs in c
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    u = c[i]
                    v = c[j]
                    edge = (min(u, v), max(u, v))
                    if edge in edge_to_idx:
                        A[edge_to_idx[edge], c_idx] = 1
                    else:
                        # This shouldn't happen if candidates are valid cliques
                        pass
                        
        # Objective: minimize sum(x)
        c_obj = np.ones(num_candidates)
        
        # Constraints: sum_{c containing e} x_c = 1 for all e
        # A x = 1
        # Lower bound = 1, Upper bound = 1
        
        cons = [LinearConstraint(A, 1, 1)]
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x is binary
        integrality = np.ones(num_candidates)
        
        # Solve
        try:
            res = milp(c=c_obj, constraints=cons, integrality=integrality, bounds=bounds)
            
            if not res.success:
                # Fallback to greedy or trivial
                # Trivial: each edge is a clique
                return [[u, v] for u, v in edge_set]
                
            # Extract solution
            x = res.x
            solution_cliques = []
            for i in range(num_candidates):
                if x[i] > 0.5:
                    solution_cliques.append(unique_candidates[i])
                    
            # Verify solution
            if not verify_partition(solution_cliques, edge_set):
                # If ILP failed to provide a valid partition (due to numerical issues or infeasibility of exact cover with these candidates),
                # we need a fallback.
                # The exact cover might be infeasible with the restricted candidate set.
                # In that case, we should expand candidates or use a different method.
                # For now, fallback to greedy.
                return greedy_partition(n, k, edges, adj)
                
            return solution_cliques
            
        except Exception:
            return greedy_partition(n, k, edges, adj)

    def verify_partition(partition, edge_set):
        # Check if every edge is in exactly one clique
        covered_count = np.zeros(len(edge_set))
        edge_to_idx = {}
        for idx, e in enumerate(edge_set):
            edge_to_idx[e] = idx
            
        for clique in partition:
            # Check if clique is actually a clique in the graph? 
            # We assumed candidates are cliques, so yes.
            # But we should check if the clique has >= 2 vertices.
            if len(clique) < 2:
                return False
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u = clique[i]
                    v = clique[j]
                    edge = (min(u, v), max(u, v))
                    if edge in edge_to_idx:
                        covered_count[edge_to_idx[edge]] += 1
                    else:
                        # Edge not in graph? Invalid clique.
                        return False
                        
        # All counts must be exactly 1
        if not np.all(covered_count == 1):
            return False
            
        return True

    def greedy_partition(n, k, edges, adj):
        # Simple greedy fallback
        uncovered_edges = set(edges)
        partition = []
        
        # Generate maximal cliques again
        max_cliques = []
        if k > 1:
            max_cliques.append(list(range(k)))
        for v in range(k, n):
            neighbors = [u for u in range(k) if u in adj[v]]
            if neighbors:
                max_cliques.append([v] + neighbors)
                
        while uncovered_edges:
            best_subset = []
            best_size = 0
            
            for M in max_cliques:
                # Find largest subset of M with all edges uncovered
                # Greedy max clique on subgraph of M induced by uncovered edges
                cand_list = list(M)
                # Sort by degree in uncovered subgraph
                def local_deg(u):
                    return sum(1 for w in cand_list if u != w and (min(u,w), max(u,w)) in uncovered_edges)
                
                cand_list.sort(key=local_deg, reverse=True)
                
                selected = []
                for u in cand_list:
                    is_conn = True
                    for s in selected:
                        if (min(u,s), max(u,s)) not in uncovered_edges:
                            is_conn = False
                            break
                    if is_conn:
                        selected.append(u)
                        
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_subset = selected
                    
            if not best_subset:
                e = next(iter(uncovered_edges))
                best_subset = list(e)
                
            if len(best_subset) >= 2:
                partition.append(best_subset)
                for i in range(len(best_subset)):
                    for j in range(i+1, len(best_subset)):
                        u, v = best_subset[i], best_subset[j]
                        uncovered_edges.discard((min(u,v), max(u,v)))
                        
        return partition

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph_ilp(*spec)
            # Ensure types are correct
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                if len(clean_clique) >= 2:
                    clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
# EVOLVE-BLOCK-END
