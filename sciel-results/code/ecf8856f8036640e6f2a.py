"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def find_maximal_cliques(adj_matrix, n):
        """
        Find all maximal cliques in a graph.
        Since the graph is chordal (split), we can use a simpler approach or Bron-Kerbosch.
        For small n (<=90), Bron-Kerbosch with pivoting is fast.
        However, we can exploit the structure: C is a clique.
        Maximal cliques are either:
        1. C itself (if no I vertex is connected to all of C? No, C is always a clique).
           Actually, any subset of C is a clique. But maximal cliques involving I vertices
           are of the form {v} U N(v) where v in I.
           Also, C itself is a clique. Is C maximal?
           C is maximal if there is no v in I such that N(v) == C.
           If there is v in I with N(v) == C, then {v} U C is a maximal clique (size k+1).
           Otherwise, C is a maximal clique.
           
           Wait, are there other maximal cliques?
           In a split graph, every maximal clique is either contained in C or contains exactly one vertex from I.
           Proof: If a clique contains two vertices u, v from I, they are not adjacent (I is independent). Contradiction.
           So maximal cliques are:
           - Subsets of C that are not extendable to a larger clique.
           - Sets {v} U N(v) for v in I, if N(v) is not contained in another N(v') U {v'}?
             Actually, {v} U N(v) is a clique. It is maximal if there is no w in I (w!=v) such that
             {v} U N(v) is a subset of {w} U N(w).
             Since v and w are in I, they are not adjacent. So {v} U N(v) cannot be a subset of {w} U N(w) 
             unless v is in N(w) or w is in N(v)? No, v is not in N(w) because N(w) subset C.
             So {v} U N(v) and {w} U N(w) are incomparable as sets of vertices?
             Let's check: v is in the first set, not the second (since second set is {w} U subset(C)).
             w is in the second set, not the first.
             So they are never subsets of each other.
             Thus, every {v} U N(v) is a maximal clique?
             Not necessarily. It could be that N(v) is not a clique? No, N(v) is a subset of C, which is a clique.
             So {v} U N(v) is always a clique.
             Is it maximal? Yes, because you can't add any other vertex.
             - Can't add another I vertex (independent).
             - Can't add a C vertex not in N(v) (not adjacent to v).
             
           So the set of maximal cliques is:
           1. {v} U N(v) for each v in I.
           2. C itself, IF there is no v in I such that N(v) == C.
              If there is a v with N(v) == C, then C is a subset of {v} U C, so C is not maximal.
              If there are multiple v with N(v) == C, C is still not maximal.
              
           Let's verify.
           Graph: C={0,1}, I={2,3}. Edges: (0,1), (0,2), (1,2), (0,3).
           N(2) = {0,1} = C. Maximal clique: {0,1,2}.
           N(3) = {0}. Maximal clique: {0,3}.
           Is C={0,1} maximal? No, subset of {0,1,2}.
           Are there any other maximal cliques?
           {0,1} is covered. {0,3} is covered.
           What about {1}? Subset of {0,1,2}.
           
           So the list of maximal cliques is:
           M = [ {v} U N(v) for v in I ]
           If not any(N(v) == C for v in I):
               M.append(C)
           
           This is extremely efficient! O(n * k).
        """
        maximal_cliques = []
        
        # Identify C and I
        C = set(range(k))
        I = set(range(k, n))
        
        # Compute N(v) for v in I
        N_v = {}
        for v in I:
            neighbors = set()
            for u in C:
                if adj_matrix[u, v]:
                    neighbors.add(u)
            N_v[v] = neighbors
            
        # Check if C is a subset of any N(v) U {v}
        # i.e., is there a v such that N(v) == C?
        c_is_maximal = True
        for v in I:
            if N_v[v] == C:
                c_is_maximal = False
                break
                
        # Add cliques for I vertices
        for v in I:
            if N_v[v]: # If N(v) is empty, {v} is a clique of size 1, but we need cliques of size >= 2 for edge cover?
                       # Actually, the problem says "Every listed set must be a clique with at least two vertices".
                       # So we only care about cliques of size >= 2.
                       # If N(v) is empty, {v} has size 1, ignore.
                       # If N(v) has size 1, {v, u} has size 2, valid.
                       # If N(v) has size >= 2, {v} U N(v) has size >= 3, valid.
                clique = list(N_v[v]) + [v]
                # Ensure it's a clique (it is by construction)
                # Ensure size >= 2
                if len(clique) >= 2:
                    maximal_cliques.append(clique)
        
        # Add C if maximal and size >= 2
        if c_is_maximal and len(C) >= 2:
            maximal_cliques.append(list(C))
            
        # Note: Are there any other maximal cliques?
        # We established that any maximal clique is either contained in C or contains exactly one I vertex.
        # If it contains an I vertex v, it must be a subset of {v} U N(v).
        # Since {v} U N(v) is a clique, any maximal clique containing v must be {v} U N(v) itself?
        # Yes, because you can't add more vertices.
        # What if a maximal clique is contained in C?
        # Then it must be C itself (since C is a clique, any proper subset is not maximal).
        # So the list is complete.
        
        return maximal_cliques

    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        # Find maximal cliques
        max_cliques = find_maximal_cliques(adj, n)
        
        if not max_cliques:
            # Should not happen if edges exist, but fallback
            return [[u, v] for u, v in edges]
            
        # Map edges to indices
        edge_list = edges
        edge_index = {e: i for i, e in enumerate(edge_list)}
        num_edges = len(edge_list)
        
        # For each maximal clique, find which edges it covers
        # A clique S covers edge (u,v) if u in S and v in S
        clique_edges = []
        for clique in max_cliques:
            clique_set = set(clique)
            covered_edges = []
            # Iterate over pairs in clique
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if (u, v) in edge_index:
                        covered_edges.append(edge_index[(u, v)])
                    elif (v, u) in edge_index:
                        covered_edges.append(edge_index[(v, u)])
            if covered_edges:
                clique_edges.append(covered_edges)
                
        # It's possible some edges are not covered by any maximal clique?
        # No, every edge is in some maximal clique.
        # Let's verify coverage
        covered = [False] * num_edges
        for ce in clique_edges:
            for idx in ce:
                covered[idx] = True
        if not all(covered):
            # Fallback to edge cliques if something went wrong
            return [[u, v] for u, v in edges]
            
        num_cliques = len(clique_edges)
        
        # Formulate MILP
        # Variables x_j for each maximal clique j
        # Minimize sum x_j
        # Subject to: for each edge i, sum_{j: i in clique_edges[j]} x_j == 1
        # x_j in {0, 1}
        
        # Cost vector
        c = np.ones(num_cliques)
        
        # Constraint matrix A
        # A[i, j] = 1 if edge i is in clique j
        A = np.zeros((num_edges, num_cliques))
        for j, ce in enumerate(clique_edges):
            for i in ce:
                A[i, j] = 1
        
        # Constraints: A x = 1
        # LinearConstraint(A, 1, 1)
        constraints = LinearConstraint(A, 1, 1)
        
        # Bounds
        bounds = Bounds(0, 1)
        
        # Integrality
        integrality = np.ones(num_cliques)
        
        # Solve
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # Extract solution
                x = res.x
                # Due to floating point, threshold
                x = (x > 0.5).astype(int)
                
                # Construct cliques
                solution_cliques = []
                for j in range(num_cliques):
                    if x[j] == 1:
                        solution_cliques.append(max_cliques[j])
                        
                # Validate
                if len(solution_cliques) > 0:
                    # Check coverage
                    covered = [False] * num_edges
                    for clique in solution_cliques:
                        c_set = set(clique)
                        for i, (u, v) in enumerate(edge_list):
                            if u in c_set and v in c_set:
                                covered[i] = True
                    if all(covered):
                        # Convert to lists of ints
                        return [[int(v) for v in clique] for clique in solution_cliques]
                    
                # If MILP failed or solution invalid, fallback to greedy or edges
            else:
                pass # Fallback
        except Exception:
            pass # Fallback

        # Fallback: Greedy or Edge Cliques
        # Since MILP might be slow or fail, we need a robust fallback.
        # The greedy in the original code was okay but not optimal.
        # Let's use the greedy from the original code as a safety net, 
        # but since we changed the structure, let's just return edge cliques 
        # if MILP fails, as it's guaranteed valid.
        # However, edge cliques score is bad.
        # Let's implement a quick greedy fallback here.
        
        def greedy_partition():
            available_cc = set()
            for u in range(k):
                for v in range(u+1, k):
                    available_cc.add((u, v))
            
            E_CI = {}
            for u, v in edges:
                if u < k and v >= k:
                    if v not in E_CI:
                        E_CI[v] = set()
                    E_CI[v].add(u)
                elif v < k and u >= k:
                    if u not in E_CI:
                        E_CI[u] = set()
                    E_CI[u].add(v)
            
            I_vertices = list(E_CI.keys())
            order = sorted(I_vertices, key=lambda v: len(E_CI[v]), reverse=True)
            
            cliques = []
            assigned_ci = {}
            
            for v in order:
                if v not in E_CI or not E_CI[v]:
                    continue
                neighbors = list(E_CI[v])
                
                def degree_in_avail(u, cand_list):
                    cnt = 0
                    for w in cand_list:
                        if u != w and (min(u, w), max(u, w)) in available_cc:
                            cnt += 1
                    return cnt
                
                candidates = list(neighbors)
                candidates.sort(key=lambda x: degree_in_avail(x, candidates), reverse=True)
                
                S = []
                while candidates:
                    u = candidates[0]
                    S.append(u)
                    new_cands = []
                    for w in candidates[1:]:
                        if (min(u, w), max(u, w)) in available_cc:
                            new_cands.append(w)
                    candidates = new_cands
                
                if len(S) >= 2:
                    cliques.append(S + [v])
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            available_cc.discard((min(S[i], S[j]), max(S[i], S[j])))
                    assigned_ci[v] = set(S)
                elif len(S) == 1:
                    cliques.append([S[0], v])
                    assigned_ci[v] = set(S)
            
            for v in E_CI:
                assigned = assigned_ci.get(v, set())
                for u in E_CI[v]:
                    if u not in assigned:
                        cliques.append([u, v])
            
            for edge in available_cc:
                cliques.append([edge[0], edge[1]])
                
            return cliques

        try:
            return greedy_partition()
        except Exception:
            return [[u, v] for u, v in edges]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[int(u), int(v)] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)

    return results
# EVOLVE-BLOCK-END
