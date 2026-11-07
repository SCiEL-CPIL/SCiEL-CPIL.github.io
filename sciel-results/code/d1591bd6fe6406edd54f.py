"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

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

    def solve_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # Build adjacency and edge sets
        # Vertices 0..k-1 are C, k..n-1 are I
        adj = {i: set() for i in range(n)}
        edge_set = set(edges)
        
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Identify edges in C (internal) and C-I (cross)
        edges_cc = []
        edges_ci = []
        for u, v in edges:
            if u < k and v < k:
                edges_cc.append((u, v))
            else:
                # In split graph, one is in C, one in I
                edges_ci.append((u, v))
        
        edges_cc_set = set(edges_cc)
        edges_ci_set = set(edges_ci)
        all_edges = edges_cc_set | edges_ci_set
        edge_index = {e: i for i, e in enumerate(all_edges)}

        # Generate Candidate Cliques
        # Strategy:
        # 1. All maximal cliques? In a split graph, maximal cliques are of the form S U {v} where S subset C, v in I, and S = N(v) intersect C?
        #    Actually, a maximal clique containing v in I is N(v) U {v}.
        #    A maximal clique not containing any I vertex is just C (the whole clique).
        #    However, we might want smaller cliques if maximal ones overlap too much or if we need to partition edges.
        #    For ILP, we need a set of candidates that can partition the edges.
        #    
        #    Key Insight: Any valid clique partition can be represented using cliques.
        #    We should include:
        #    - The full clique C (covers all CC edges).
        #    - For each v in I, the clique N(v) U {v} (covers all CI edges for v and some CC edges).
        #    - Also, to allow flexibility, we should include "strips" or smaller cliques?
        #    If we only use maximal cliques, can we always find a partition?
        #    Not necessarily. But for split graphs, the structure is simple.
        #    
        #    Let's generate a richer set:
        #    - C (the clique on C)
        #    - For each v in I: N(v) U {v}
        #    - For each v in I: {u, v} for all u in N(v) (size 2 cliques)
        #    - For each v in I: {u1, u2, v} for all pairs u1, u2 in N(v) such that (u1, u2) is an edge in C (always true)
        #      This might be too many. k=30, |N(v)|~21. Pairs ~ 200. 60 vertices * 200 = 12000 candidates.
        #      Plus C.
        #      This is manageable for MILP.
        
        candidates = []
        
        # 1. The full clique C
        candidates.append(list(range(k)))
        
        # 2. For each v in I
        for v in range(k, n):
            Nv = [u for u in range(k) if (min(u, v), max(u, v)) in edge_set]
            if not Nv:
                continue
            
            # 2a. Maximal clique for v: N(v) U {v}
            candidates.append(Nv + [v])
            
            # 2b. Size 2 cliques {u, v}
            for u in Nv:
                candidates.append([u, v])
            
            # 2c. Size 3 cliques {u1, u2, v}
            # Only include if it helps? It covers 1 CC edge and 2 CI edges.
            # It's a "better" trade-off than two size-2 cliques (which cover 2 CI edges and 0 CC edges)
            # IF the CC edge is not needed elsewhere.
            # To keep ILP small, let's include all size 3 cliques for v.
            # Number of pairs in Nv is ~ k^2/4. For k=30, ~225.
            # Total candidates ~ 60 * (1 + 21 + 225) ~ 14000.
            # Variables: 14000. Constraints: ~1700 edges.
            # This is a small ILP.
            
            for i in range(len(Nv)):
                for j in range(i+1, len(Nv)):
                    u1, u2 = Nv[i], Nv[j]
                    # Edge (u1, u2) is always in C
                    candidates.append([u1, u2, v])
            
            # 2d. Size 4 cliques? {u1, u2, u3, v}
            # Covers 3 CI edges and 3 CC edges.
            # Number of triples is ~ k^3/6. 30^3/6 = 4500.
            # 60 * 4500 = 270,000. Might be too slow to build or solve.
            # Let's stick to size 3 for now. If time permits, we could add size 4.
            # Given the deadline, size 3 is a good balance.

        # Remove duplicates (convert to frozenset for uniqueness, but keep list for output)
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = frozenset(c)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        num_candidates = len(unique_candidates)
        num_edges = len(all_edges)
        
        # Setup ILP
        # Variables x_j >= 0 integer, for j in 0..num_candidates-1
        # Minimize sum x_j
        # Subject to: For each edge e, sum_{j: e in candidate_j} x_j = 1
        
        # Build constraint matrix A (sparse)
        # A[i, j] = 1 if edge i is in candidate j
        # We use scipy sparse matrix
        
        import scipy.sparse as sp
        
        row_idx = []
        col_idx = []
        data = []
        
        for j, cand in enumerate(unique_candidates):
            for u in cand:
                for v in cand:
                    if u >= v: continue
                    e = (min(u, v), max(u, v))
                    if e in edge_index:
                        i = edge_index[e]
                        row_idx.append(i)
                        col_idx.append(j)
                        data.append(1)
        
        A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(num_edges, num_candidates))
        
        # Objective: minimize sum x
        c = np.ones(num_candidates)
        
        # Constraints: A x = 1
        # LinearConstraint(A, 1, 1)
        cons = LinearConstraint(A, 1, 1)
        
        # Bounds: 0 <= x <= 1 (since we partition, we don't need x > 1, and 0/1 is optimal for set partitioning usually, but integer is enough)
        bounds = Bounds(0, 1)
        
        # Integrality: all integer
        integrality = np.ones(num_candidates)
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                # Extract solution
                selected_cliques = []
                for j in range(num_candidates):
                    if res.x[j] > 0.5:
                        selected_cliques.append(unique_candidates[j])
                
                # Verify validity (just in case)
                # In a real scenario, we'd check, but for speed, we trust MILP.
                # However, we must ensure types are Python int.
                final_cliques = []
                for c_list in selected_cliques:
                    final_cliques.append([int(v) for v in c_list])
                
                return final_cliques
            else:
                # Fallback to greedy if ILP fails
                return _greedy_fallback(n, k, edges)
                
        except Exception as e:
            # Fallback to greedy if ILP crashes or times out
            return _greedy_fallback(n, k, edges)

    def _greedy_fallback(n, k, edges):
        # Simple greedy: Max clique per vertex
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edges_cc_set = set((u, v) for u, v in edges if u < k and v < k)
        edges_ci_set = set((u, v) for u, v in edges if not (u < k and v < k))
        
        cliques = []
        
        # Process I vertices
        for v in range(k, n):
            Nv = [u for u in range(k) if u in adj[v]]
            if not Nv: continue
            
            # Greedy max clique in Nv respecting remaining CC edges
            S = []
            for u in Nv:
                can_add = True
                for u2 in S:
                    e = (min(u, u2), max(u, u2))
                    if e not in edges_cc_set:
                        can_add = False
                        break
                if can_add:
                    S.append(u)
            
            if S:
                cliques.append(S + [v])
                for u in S:
                    edges_ci_set.discard((min(u, v), max(u, v)))
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        edges_cc_set.discard((min(S[i], S[j]), max(S[i], S[j])))
                        
            for u in Nv:
                e = (min(u, v), max(u, v))
                if e in edges_ci_set:
                    cliques.append([u, v])
                    edges_ci_set.discard(e)
        
        # Remaining CC
        for u, v in edges_cc_set:
            cliques.append([u, v])
            
        # Remaining CI (should be empty)
        for u, v in edges_ci_set:
            cliques.append([u, v])
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed-like solution (size 2 cliques)
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            # Ensure types are correct
            clean_res = []
            for c in res:
                clean_res.append([int(v) for v in c])
            results.append(clean_res)
            
    return results
# EVOLVE-BLOCK-END
