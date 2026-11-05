import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_pairs = len(all_pairs)
    
    # 1. Build the conflict graph of edges
    # Nodes are the edges of the complete graph K_20.
    # Two edge-nodes are connected if selecting both creates a K_{2,3} subgraph.
    # A K_{2,3} is formed by two edges (u,v) and (x,y) such that they share 3 common neighbors.
    # Actually, the definition is: no two vertices have 3 common neighbors.
    # This is equivalent to saying the graph is K_{2,3}-free.
    # In the conflict graph for MIS:
    # We want to select a set of edges S such that for any u, v, |N(u) ∩ N(v)| <= 2.
    # If we add edge (u,v) to a set S, it conflicts with any existing edge (x,y) in S
    # if adding (u,v) would cause some pair of vertices to have >2 common neighbors.
    
    # Let's precompute conflicts between any two edges e1=(u,v) and e2=(x,y).
    # If we have a set of edges, the constraint is global.
    # However, for Branch and Bound on Independent Set, we need a static conflict graph.
    # Two edges e1 and e2 conflict if they cannot both be in a K_{2,3}-free graph?
    # No, that's not quite right. Two edges can be in a K_{2,3}-free graph together, 
    # but they might conflict with a third edge.
    # Wait, the property "no two vertices have 3 common neighbors" is not pairwise compatible in a simple way for all triples.
    # Actually, it IS pairwise in a specific sense?
    # Let's check: If we have edges e1 and e2. Do they conflict?
    # If we select e1 and e2, is it possible to select a third edge e3 that makes it invalid? Yes.
    # So the "conflict graph" for MIS formulation requires that a set of edges is an independent set in the conflict graph if and only if it is a valid K_{2,3}-free graph.
    # This implies that any subset of a valid set is valid (hereditary property).
    # K_{2,3}-freeness is hereditary.
    # However, the conflict graph for MIS usually encodes pairwise incompatibilities.
    # If a set {e1, e2, e3} is invalid, but {e1, e2} is valid and {e1, e3} is valid, etc., then pairwise conflict graphs are not sufficient to represent the constraint directly as an Independent Set problem unless the constraint is 2-wise.
    # Is K_{2,3}-freeness 2-wise?
    # If I have edges e1 and e2, can I always check if they are "compatible"?
    # If e1=(u,v) and e2=(x,y).
    # If I pick both, do they create a K_{2,3}? No, a K_{2,3} requires 5 vertices (2 in one part, 3 in other) and 6 edges.
    # Two edges cannot form a K_{2,3}.
    # So any pair of edges is compatible.
    # This means the conflict graph has NO edges!
    # Therefore, MIS on the edge-conflict graph is trivial (select all edges).
    # 
    # My previous thought process in the prompt description was slightly flawed or I was thinking of a different formulation.
    # The constraint is on triples or larger sets.
    # Specifically, for any pair of vertices (u,v), the number of common neighbors must be <= 2.
    # This is a constraint on the *degrees* in the common-neighbor matrix.
    # 
    # So, a standard MIS on pairwise conflicts doesn't work directly because the conflicts are not pairwise between edges.
    # 
    # Alternative Strategy: Branch and Bound on the edge set, but using a different bound.
    # Or, use the fact that we can model this as:
    # Maximize sum(x_e) subject to for all u != v: sum_{w != u,v} x_{uw} x_{vw} <= 2.
    # This is a quadratic binary optimization problem.
    # 
    # Since the prompt specifically suggested "Model the problem as a maximum independent set on the conflict graph... where two edges conflict if they form a K2,3", this premise is mathematically incorrect for pairwise edges. 
    # However, maybe it meant "conflict graph of vertices"? No.
    # 
    # Let's re-read the directive carefully: "Model the problem as a maximum independent set on the conflict graph of all C(20,2) edges, where two edges conflict if they form a K2,3 subgraph".
    # If I follow this literally, the conflict graph is empty, and the answer is C(20,2)=190, which is wrong (K20 contains K2,3).
    # 
    # Perhaps the intended meaning is: We are building the graph edge by edge.
    # When considering adding edge e=(u,v), it conflicts with the *current state* if it violates the constraint.
    # This is a sequential decision process.
    # 
    # Let's try a different structural change: **Greedy with Lookahead / Backtracking**.
    # Or, **Local Search with a better objective**.
    # 
    # Actually, there is a known structure for K_{2,3}-free graphs.
    # The extremal number ex(n, K_{2,3}) is known to be roughly n^{3/2}.
    # For n=20, n^{3/2} = 89.
    # Current score 2.0 implies value ~ 2 * seed. If seed is ~45, value is ~90.
    # 
    # Let's try a **Deterministic Greedy with Randomized Restart and Local Repair**.
    # The current program uses SA. SA can get stuck.
    # A purely greedy approach that picks the edge that minimizes the "tightness" of the constraint might be more robust.
    # 
    # Strategy: 
    # 1. Iterate through edges in a specific order (e.g., random or weighted).
    # 2. Add edge if valid.
    # 3. After full pass, try to "swap" edges to improve density?
    # 
    # Better: **Simulated Annealing with a Stronger Penalty and Adaptive Cooling**.
    # The current SA might be too slow or the penalty too weak.
    # 
    # Let's try **Branch and Bound** properly.
    # We can bound the number of additional edges we can add.
    # If we have a partial graph G, and we want to add more edges.
    # For each remaining pair (u,v), check if it can be added.
    # This is just greedy.
    # 
    # To get a global optimum, we need to explore.
    # 
    # Let's try **Local Search (Hill Climbing) with Multiple Restarts and a Better Move Set**.
    # Moves: Add edge, Remove edge, Swap edge (remove one, add another).
    # 
    # Actually, let's look at the "Inspiration 3": "deterministic greedy construction based on a fixed vertex ordering and local saturation".
    # This scored 1.8667.
    # Current is 2.0000.
    # 
    # Let's try a **Genetic Algorithm** or **Memetic Algorithm**.
    # 
    # Given the time limit (120s), and the need for a structural change:
    # I will implement a **Fast Local Search** using NumPy for vectorized operations to allow more iterations.
    # The bottleneck in the current code is the Python loops for `compute_cn` and `get_penalty`.
    # By using NumPy, we can evaluate many moves much faster.
    # 
    # **New Strategy: Vectorized Simulated Annealing / Hill Climbing using NumPy bitsets.**
    # Represent the adjacency matrix as a NumPy array of ints.
    # Compute Common Neighbors (CN) using matrix multiplication: `cn = adj @ adj`.
    # `cn[u][v]` is the number of common neighbors.
    # Constraint: `cn[u][v] <= 2` for all `u != v`.
    # 
    # This allows O(n^3) matrix mult for full CN update, but we can do incremental updates.
    # Incremental update of CN when adding edge (u,v):
    # `cn[u, :] += adj[v, :]`
    # `cn[v, :] += adj[u, :]`
    # `cn[:, u] += adj[:, v]`
    # `cn[:, v] += adj[:, u]`
    # And fix the diagonal/zero entries if needed, or just track them.
    # Actually, `cn[u,v]` counts w such that adj[u,w] and adj[v,w].
    # If we add edge (u,v), we don't change adj[u,w] or adj[v,w] for w != u,v.
    # But we change adj[u,v] and adj[v,u].
    # Does this affect CN?
    # CN[u][w] for w != u,v: depends on adj[u, *] and adj[w, *].
    # If we add edge (u,v), adj[u,v] becomes 1.
    # This affects CN[u][v] (becomes 1 + old common neighbors? No, CN[u][v] counts w != u,v. So adj[u,v] itself is not counted in CN[u][v] if we define CN as common neighbors excluding the vertices themselves? 
    # Standard definition: N(u) is set of neighbors. |N(u) ∩ N(v)|.
    # If u and v are adjacent, is u in N(v)? Yes. Is v in N(u)? Yes.
    # So if we add edge (u,v), u enters N(v) and v enters N(u).
    # Does u enter N(u) ∩ N(v)? u is in N(v). Is u in N(u)? No (simple graph).
    # So the common neighbors of u and v are vertices w such that w~u and w~v.
    # Adding edge (u,v) does NOT change the set of common neighbors w.
    # It only changes the adjacency of u and v to each other.
    # So `cn[u][v]` (count of w != u,v) remains unchanged by adding edge (u,v).
    # 
    # Wait, let's verify.
    # G1: u-w, v-w. w is common neighbor.
    # Add u-v.
    # G2: u-w, v-w, u-v.
    # Common neighbors of u and v in G2: w is still a common neighbor.
    # Are there new common neighbors?
    # A vertex x is a common neighbor if x~u and x~v.
    # Adding u-v doesn't create new edges from u or v to other vertices.
    # So the set of common neighbors is invariant under adding an edge between them?
    # YES.
    # 
    # So, if we add edge (u,v), the CN matrix entries `cn[u][v]` and `cn[v][u]` do NOT change.
    # But what about `cn[u][w]` for w != u,v?
    # `cn[u][w]` counts x such that x~u and x~w.
    # Adding edge (u,v) makes v a neighbor of u.
    # So v becomes a common neighbor of u and w IF v~w.
    # So if adj[v][w] is 1, then `cn[u][w]` increases by 1.
    # Similarly, `cn[v][w]` increases by 1 if adj[u][w] is 1.
    # 
    # So the update rules are:
    # When adding edge (u,v):
    # 1. `adj[u,v] = 1`, `adj[v,u] = 1`.
    # 2. For all w != u,v:
    #    if `adj[v,w]`: `cn[u,w] += 1`, `cn[w,u] += 1`
    #    if `adj[u,w]`: `cn[v,w] += 1`, `cn[w,v] += 1`
    # 
    # This is O(n) per edge addition.
    # 
    # Using NumPy, we can vectorize the check for validity.
    # Validity check for adding (u,v):
    # We need to ensure that for all w, the new `cn[u,w]` and `cn[v,w]` do not exceed 2.
    # New `cn[u,w]` = `cn[u,w]` + (1 if `adj[v,w]` else 0).
    # So we need: `cn[u,w] + adj[v,w] <= 2` for all w.
    # And `cn[v,w] + adj[u,w] <= 2` for all w.
    # 
    # This can be checked in O(n) or vectorized O(1) with NumPy if we precompute.
    # 
    # Let's implement a fast SA using NumPy.
    
    # Initialize
    adj = np.zeros((n, n), dtype=np.int8)
    cn = np.zeros((n, n), dtype=np.int16)
    
    # Helper to compute cn from scratch (for initialization)
    def compute_cn_np(adj):
        # cn = adj @ adj
        # But this counts u in N(u)? No, simple graph adj[u,u]=0.
        # adj @ adj gives number of paths of length 2.
        # (adj @ adj)[u][v] = sum_w adj[u][w] * adj[w][v].
        # This is exactly the number of common neighbors.
        return (adj @ adj).astype(np.int16)
    
    # Helper to check if edge (u,v) can be added
    def can_add(u, v):
        # Check constraints for u and v
        # For all w: cn[u,w] + adj[v,w] <= 2
        # For all w: cn[v,w] + adj[u,w] <= 2
        # Note: adj[v,u] is 0 currently (if adding).
        # We are checking if adding (u,v) violates constraint.
        
        # Vectorized check
        # mask_u = adj[v, :]
        # new_cn_u = cn[u, :] + mask_u
        # if np.any(new_cn_u > 2): return False
        
        # But we must exclude u and v themselves?
        # cn[u,u] is 0. adj[v,u] is 0. So cn[u,u]+0 = 0 <= 2. OK.
        # cn[u,v] + adj[v,v] (0) = cn[u,v]. 
        # Wait, if we add (u,v), does cn[u,v] change? No.
        # So the constraint on cn[u,v] is just that it must be <= 2.
        # But the formula `cn[u,w] + adj[v,w]` for w=v gives `cn[u,v] + adj[v,v] = cn[u,v]`.
        # So it correctly checks the existing cn[u,v].
        
        # However, we must ensure that we don't count the edge (u,v) itself in the common neighbors?
        # The definition of K2,3 free is about common neighbors.
        # The edge (u,v) is not a common neighbor.
        # So the logic holds.
        
        # One subtlety: if u and v are already connected? We only call this if not.
        
        # Check 1: for all w, cn[u,w] + adj[v,w] <= 2
        # Check 2: for all w, cn[v,w] + adj[u,w] <= 2
        
        # We can do this in one go?
        # Let's just loop w for speed in Python? No, NumPy is faster for n=20?
        # n=20 is small. Python loop might be faster than NumPy overhead.
        # Let's use Python loop for the check, it's only 18 iterations.
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                if cn[u, w] >= 2:
                    return False
            if adj[u, w]:
                if cn[v, w] >= 2:
                    return False
        return True

    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                cn[u, w] += 1
                cn[w, u] += 1
            if adj[u, w]:
                cn[v, w] += 1
                cn[w, v] += 1

    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                cn[u, w] -= 1
                cn[w, u] -= 1
            if adj[u, w]:
                cn[v, w] -= 1
                cn[w, v] -= 1

    def get_penalty():
        # Sum of max(0, cn[u,v] - 2)
        # Vectorized
        # Mask out diagonal
        pen = 0
        for u in range(n):
            for v in range(u+1, n):
                if cn[u, v] > 2:
                    pen += cn[u, v] - 2
        return pen

    # Main Loop: SA with fast checks
    best_m = 0
    best_adj = adj.copy()
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Initialize randomly
        adj = np.zeros((n, n), dtype=np.int8)
        cn = np.zeros((n, n), dtype=np.int16)
        
        # Random initial edges
        p = 0.3
        for u, v in all_pairs:
            if random.random() < p:
                add_edge(u, v)
        
        m = np.sum(adj) // 2
        penalty = get_penalty()
        
        best_m_in_run = m
        best_adj_in_run = adj.copy()
        
        T = 5.0
        cooling = 0.99
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 5:
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                u, v = random.choice(all_pairs)
                is_adding = adj[u, v] == 0
                
                # Calculate delta penalty
                delta_pen = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # If adding, cn[u,w] increases if adj[v,w]
                    # If removing, cn[u,w] decreases if adj[v,w]
                    delta = 1 if is_adding else -1
                    if adj[v, w]:
                        old_cn = cn[u, w]
                        new_cn = old_cn + delta
                        delta_pen += max(0, new_cn - 2) - max(0, old_cn - 2)
                    if adj[u, w]:
                        old_cn = cn[v, w]
                        new_cn = old_cn + delta
                        delta_pen += max(0, new_cn - 2) - max(0, old_cn - 2)
                
                delta_m = 1 if is_adding else -1
                delta_score = delta_m - 10 * delta_pen
                
                if delta_score > 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(delta_score / T)
                
                if accept:
                    if is_adding:
                        add_edge(u, v)
                    else:
                        remove_edge(u, v)
                    
                    m += delta_m
                    penalty += delta_pen
                    
                    if m > best_m_in_run:
                        best_m_in_run = m
                        best_adj_in_run = adj.copy()
            
            T *= cooling
        
        if best_m_in_run > best_m:
            best_m = best_m_in_run
            best_adj = best_adj_in_run

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return edges

