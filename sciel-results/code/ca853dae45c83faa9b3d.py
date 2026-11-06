"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    # Total possible edges
    total_edges = n * (n - 1) // 2
    
    # Map edge index to (u, v)
    edge_map = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
    
    # Build incidence matrix: rows = edges, cols = vertices
    # A[i, j] = 1 if edge i is incident to vertex j
    A = np.zeros((total_edges, n))
    for i, (u, v) in enumerate(edge_map):
        A[i, u] = 1
        A[i, v] = 1
    
    # Identify all K2,3 subgraphs
    # A K2,3 is formed by two vertices u, v (the "pole") and three vertices w1, w2, w3 (the "trunk")
    # such that all edges (u, wi), (v, wi) exist.
    # Constraint: For any such set of 6 edges, at most 5 can be present.
    # i.e., sum(x_e for e in E_K23) <= 5
    
    # To be efficient, we generate constraints for all potential K2,3s.
    # Number of pairs (u,v) is C(20,2) = 190.
    # For each pair, common neighbors can be any 3 from the remaining 18.
    # C(18, 3) = 816.
    # Total constraints ~ 190 * 816 = 155,040.
    # This might be too many for a quick solve, but let's try to generate them sparsely.
    
    # However, generating 150k constraints with 190 variables each is heavy.
    # Alternative: Use a smaller set of constraints or a different formulation?
    # Actually, scipy's milp can handle sparse matrices well.
    # Let's build the constraint matrix C where each row is a K2,3.
    
    # Optimization: We don't need to check all K2,3s if we use a heuristic approach first,
    # but the task asks for a DIFFERENT strategy. ILP is distinct.
    # Let's limit the number of constraints if it gets too slow, or just generate all.
    # 155k rows * 190 cols is ~30M non-zeros (each row has 6 non-zeros). 155k * 6 = 930k non-zeros.
    # This is very sparse and manageable.
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute neighbors for each vertex to find common neighbors quickly?
    # No, we are defining the constraints on the *variables* (edges), not the graph structure yet.
    # We need to enumerate all subsets of 3 vertices {w1, w2, w3} for each pair {u, v}.
    
    # To speed up generation, we can iterate u < v, and w1 < w2 < w3 distinct from u, v.
    # This is O(n^5) which is 20^5 = 3.2M iterations. Each iteration adds 6 non-zeros.
    # 3.2M * 6 = 19M operations. Python loops might be slow.
    
    # Let's use a more efficient enumeration or limit if necessary.
    # Actually, we can just generate the constraints.
    
    # Let's try to generate them. If it takes too long, we fall back to a heuristic.
    # But 3.2M iterations in Python might take ~1-2 seconds. Acceptable.
    
    for u in range(n):
        for v in range(u + 1, n):
            # Candidates for trunk vertices
            candidates = [w for w in range(n) if w != u and w != v]
            # Iterate over combinations of 3
            # itertools.combinations is efficient
            from itertools import combinations
            for w1, w2, w3 in combinations(candidates, 3):
                # Edges: (u, w1), (u, w2), (u, w3), (v, w1), (v, w2), (v, w3)
                # Ensure u < w_i and v < w_i for edge_map lookup?
                # edge_map is defined for u < v.
                
                def get_edge_idx(a, b):
                    if a > b:
                        a, b = b, a
                    return edge_to_idx[(a, b)]
                
                e1 = get_edge_idx(u, w1)
                e2 = get_edge_idx(u, w2)
                e3 = get_edge_idx(u, w3)
                e4 = get_edge_idx(v, w1)
                e5 = get_edge_idx(v, w2)
                e6 = get_edge_idx(v, w3)
                
                row_idx = len(row_indices)
                for e in [e1, e2, e3, e4, e5, e6]:
                    row_indices.append(row_idx)
                    col_indices.append(e)
                    data.append(1)
    
    # Build sparse matrix C
    # Rows: constraints, Cols: variables (edges)
    C = coo_matrix((data, (row_indices, col_indices)), shape=(len(row_indices), total_edges)).tocsr()
    
    # Objective: Maximize sum(x_e) -> Minimize -sum(x_e)
    c = -np.ones(total_edges)
    
    # Constraints:
    # 1. C @ x <= 5 (K2,3 free)
    # 2. 0 <= x <= 1 (binary)
    
    # milp requires:
    # c: linear objective
    # constraints: LinearConstraint(A, lb, ub)
    # integrality: 1 for binary
    # bounds: Bounds(lb, ub)
    
    # Combine constraints
    # C @ x <= 5
    # 0 <= x <= 1
    
    # Note: scipy milp minimizes c @ x subject to constraints.
    
    # We need to handle the case where generation took too long or memory is an issue.
    # But 155k constraints is fine.
    
    try:
        # Solve ILP
        # This might take a while. We set a time limit via the solver if possible?
        # scipy milp doesn't have a direct time limit param in the simple API, 
        # but we can run it in a thread or just rely on the fact that it's small.
        # Actually, for n=20, the ILP should solve quickly if the constraints are tight.
        
        # To be safe against timeout, we could run a heuristic first, then ILP.
        # But the directive asks for a different strategy. ILP is the strategy.
        # If ILP times out, we return the best found? 
        # scipy.optimize.milp doesn't support callback for early stopping easily.
        
        # Let's assume it solves within 120s. 
        # If not, we need a fallback.
        
        # Fallback heuristic: Greedy
        def greedy_fallback():
            adj_bits = [0] * n
            edges = []
            def can_add(adj_bits, u, v):
                common_uv = bin(adj_bits[u] & adj_bits[v]).count('1')
                if common_uv > 2:
                    return False
                nv = adj_bits[v]
                w = nv
                while w:
                    low = w & (-w)
                    wi = low.bit_length() - 1
                    w ^= low
                    if wi == u:
                        continue
                    common_uw = bin(adj_bits[u] & adj_bits[wi]).count('1')
                    if common_uw > 1:
                        return False
                nu = adj_bits[u]
                w = nu
                while w:
                    low = w & (-w)
                    wi = low.bit_length() - 1
                    w ^= low
                    if wi == v:
                        continue
                    common_vw = bin(adj_bits[v] & adj_bits[wi]).count('1')
                    if common_vw > 1:
                        return False
                return True

            all_missing = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_missing)
            for u, v in all_missing:
                if can_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append([u, v])
            return edges

        # Run ILP
        # To avoid hanging indefinitely, we rely on the small size.
        # However, if it hangs, we fail.
        # Let's add a simple timeout wrapper using threading?
        # No, standard library only. 
        # We can just try. If it's slow, the previous program was 1.96. 
        # ILP should find the optimum or near-optimum.
        
        res = milp(
            c=c,
            constraints=[
                LinearConstraint(C, ub=5),
                LinearConstraint(np.eye(total_edges), lb=0, ub=1)
            ],
            integrality=np.ones(total_edges),
            bounds=Bounds(lb=0, ub=1)
        )
        
        if res.status == 0:
            x = res.x
            edges = []
            for i, (u, v) in enumerate(edge_map):
                if x[i] > 0.5:
                    edges.append([u, v])
            return edges
        else:
            # Fallback if ILP fails
            return greedy_fallback()
            
    except Exception:
        return greedy_fallback()

# EVOLVE-BLOCK-END
