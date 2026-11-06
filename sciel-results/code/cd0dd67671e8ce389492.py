import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 20
    # Total possible edges in K_n
    total_edges = n * (n - 1) // 2
    
    # Map edge (u, v) with u < v to an index 0..total_edges-1
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
            
    # Precompute conflicts:
    # Two edges e1=(u1, v1) and e2=(u2, v2) are incompatible if including both creates a K2,3.
    # A K2,3 is formed by a pair of vertices {a, b} and 3 common neighbors.
    # If we include edge (a, b), it doesn't directly form K2,3 by itself.
    # The constraint is: for any pair of vertices {x, y}, the number of common neighbors <= 2.
    # 
    # Let's re-evaluate the conflict definition for Maximum Clique.
    # We want a set of edges E' such that G'=(V, E') is K2,3-free.
    # This is NOT a simple pairwise conflict on edges!
    # If edge e1 and e2 are "compatible" (don't form K2,3 themselves), they might still be part of a K2,3 with a third edge e3.
    # K2,3 involves 5 vertices. 
    # Wait, the constraint is "no two vertices have three common neighbors".
    # This is a global constraint on the set of edges, not a pairwise constraint on edges.
    # Therefore, Maximum Clique on a graph of edges is NOT directly applicable unless we define conflicts carefully.
    #
    # Let's check: If we have a set of edges, when is it K2,3-free?
    # For every pair {u, v}, |N(u) ∩ N(v)| <= 2.
    # This means the set of edges cannot contain a configuration that creates 3 common neighbors for any pair.
    # This is a 3-wise (or higher) interaction? No, it's a constraint on the degree of common neighbors.
    #
    # Actually, we can model this as a Maximum Independent Set in a hypergraph, or use a different approach.
    # However, the prompt suggests: "Formulate the problem as a Maximum Cardinality Independent Set (MCIS) on a conflict graph where each node represents a potential edge... and edges connect pairs of edges that would violate the K2,3 constraint if both were present."
    #
    # Does "pair of edges that would violate the K2,3 constraint if both were present" make sense?
    # If I have edge (u, v) and edge (x, y), do they violate K2,3?
    # No. A single edge doesn't create K2,3. Two edges don't create K2,3 (need 5 vertices for K2,3).
    # So pairwise conflicts are empty? That would imply all edges are compatible pairwise, which is false.
    #
    # The prompt's suggestion might be slightly flawed or I need to interpret "conflict" differently.
    # Perhaps the "nodes" are not edges, but something else?
    # Or perhaps the "conflict" is defined differently?
    #
    # Let's look at the structure of K2,3.
    # K2,3 has vertices {a, b} and {c, d, e}. Edges are (a,c), (a,d), (a,e), (b,c), (b,d), (b,e).
    # If we have these 6 edges, we have a K2,3.
    # So any subset of these 6 edges that contains all of them is invalid.
    # This is a "forbidden subgraph" problem.
    #
    # Since pairwise conflicts don't capture this, a standard Max Clique/Independent Set on edges won't work directly with simple pairwise incompatibility.
    #
    # Alternative Strategy:
    # Use a Branch and Bound search on the edges, but with a strong pruning condition based on the current common neighbor counts.
    # 
    # State: 
    # - Current set of edges (or adjacency matrix)
    # - Current edge count
    # - Potential edges to add
    # 
    # Pruning:
    # For any pair (u, v), if current_common_neighbors(u, v) >= 2, we cannot add any edge (u, w) or (v, w) where w is a common neighbor?
    # No, if common(u, v) == 2, we can still add edges, but we must ensure that adding (u, w) doesn't make common(u, v) become 3.
    # Adding (u, w) makes w a neighbor of u. If w is already a neighbor of v, then common(u, v) increases.
    # So, if common(u, v) == 2, we cannot add (u, w) or (v, w) if w is in N(u) ∩ N(v).
    # More generally, we can maintain a "capacity" for each pair.
    #
    # Let's implement a recursive Branch and Bound:
    # 1. Start with empty graph.
    # 2. Order edges by some heuristic (e.g., vertices with high degree potential).
    # 3. At each step, try to add an edge.
    # 4. If adding edge (u, v) is valid (doesn't create common > 2 for any pair), add it.
    # 5. Use upper bound: current_edges + remaining_edges.
    # 6. To improve bound, we can estimate how many more edges can be added given the current "saturated" pairs.
    #
    # This is essentially what the local search does, but B&B can be more exhaustive if the bound is good.
    # Given n=20, total edges 190. 2^190 is too big.
    # But K2,3-free graphs are sparse. Max edges ~ 100?
    # 
    # Let's try a different angle: The prompt asks for a "DIFFERENT strategy".
    # The current program uses random restarts + local search.
    # Inspiration 1 mentions "deterministic exhaustive search with symmetry breaking".
    # 
    # Let's implement a deterministic backtracking search with symmetry breaking.
    # Vertices are 0..19.
    # We can fix the degree sequence or the adjacency of vertex 0 to break symmetry?
    # Or simply use a strong upper bound.
    #
    # Upper Bound Idea:
    # For each vertex v, let d(v) be its degree.
    # Sum of d(v) = 2m.
    # Constraint: For any u, v, |N(u) ∩ N(v)| <= 2.
    # This implies that the number of paths of length 2 is limited.
    # Number of paths of length 2 = sum_v C(d(v), 2).
    # Also, number of paths of length 2 = sum_{u < v} |N(u) ∩ N(v)| <= 2 * C(n, 2) = 2 * 190 = 380.
    # So sum_v C(d(v), 2) <= 380.
    # We want to maximize m = sum d(v) / 2 subject to sum C(d(v), 2) <= 380 and d(v) <= n-1.
    # This is an integer optimization problem on degrees.
    # We can solve this LP/IP to get a tight upper bound on m.
    #
    # Let's use this degree-based upper bound in a Branch and Bound.
    
    import math
    
    def max_edges_from_degrees_bound(degrees):
        """
        Given current degrees, estimate max additional edges.
        Actually, we need a bound on the TOTAL edges given the current state.
        This is complex. Let's use a simpler bound: 
        Current edges + (n * (n-1) / 2 - current_edges) is trivial.
        
        Better bound:
        The constraint sum_v C(d(v), 2) <= 380 is a necessary condition.
        If we add edges, degrees increase.
        Let d_i be current degree of vertex i.
        Let x_i be additional degree of vertex i.
        We want to maximize sum x_i / 2.
        Subject to:
        sum_i C(d_i + x_i, 2) <= 380
        x_i <= n - 1 - d_i
        x_i >= 0
        
        This is a convex optimization problem (maximizing linear function over convex set? No, C(d,2) is convex, so the feasible region is convex. Maximizing a linear function over a convex set is easy? No, maximizing a linear function over a convex set is trivial if the set is bounded, but the optimum is at a vertex. Wait, maximizing a linear function over a convex polytope yields a vertex. But here the constraint is convex (curved). The feasible region is convex. The maximum of a linear function over a convex set is attained at an extreme point.
        
        Actually, we can just greedily allocate the "budget" of path counts to maximize edges.
        Adding 1 edge (u, v) increases C(d_u, 2) by d_u and C(d_v, 2) by d_v.
        Cost of adding edge (u, v) is d_u + d_v.
        Benefit is 1 edge.
        We have a remaining budget B = 380 - sum C(d_i, 2).
        We want to maximize number of edges added.
        This is like a knapsack problem where items are potential edges, weight is (d_u + d_v), value is 1.
        To maximize count, we should pick edges with smallest weight.
        So, sort all potential non-edges by (d_u + d_v).
        Greedily add them until budget is exceeded.
        This gives a valid upper bound (might be optimistic if it ignores vertex capacity, but it's a bound).
        """
        current_paths = sum(math.comb(d, 2) for d in degrees)
        budget = 380 - current_paths
        if budget < 0:
            return 0 # Invalid state, shouldn't happen
            
        # Potential edges
        # We need to know which edges are already present.
        # This function only takes degrees, so it's an approximation.
        # It assumes we can add any edge.
        
        # To be tighter, we should pass the adjacency matrix.
        pass

    # Let's implement the B&B with the degree bound properly.
    
    best_m = 0
    best_adj = [0] * n
    
    # Heuristic: Start with a good initial solution from local search?
    # Or just start from scratch with B&B.
    # B&B from scratch might be slow if the bound is weak.
    # Let's use the current program's local search to get a good lower bound quickly,
    # then use B&B to prove optimality or improve it?
    # The task is to maximize. If B&B is too slow, we return the best found.
    
    # To make it "MATERIALLY DIFFERENT", I will not use the local search from the seed.
    # I will use a pure deterministic B&B with symmetry breaking.
    
    # Symmetry breaking:
    # Fix vertex 0's neighbors?
    # Or fix the degree of vertex 0?
    # Let's try fixing the first few edges.
    
    # List of all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Precompute conflict sets? No, use dynamic check.
    
    adj = [0] * n
    current_m = 0
    
    # Upper bound function
    def upper_bound(adj, current_m):
        degrees = [bin(x).count('1') for x in adj]
        current_paths = sum(math.comb(d, 2) for d in degrees)
        budget = 380 - current_paths
        if budget < 0:
            return current_m
            
        # Greedy estimate of max additional edges
        # Sort potential edges by cost (d_u + d_v)
        # Note: d_u, d_v change as we add edges, so this is a static estimate.
        # It's an upper bound because we assume we can add edges with current costs.
        
        potential_costs = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    cost = degrees[u] + degrees[v]
                    potential_costs.append(cost)
                    
        potential_costs.sort()
        
        added = 0
        remaining_budget = budget
        for cost in potential_costs:
            if remaining_budget >= cost:
                remaining_budget -= cost
                added += 1
            else:
                break
                
        return current_m + added

    def dfs(idx, adj, current_m):
        nonlocal best_m, best_adj
        
        if time.time() > deadline:
            return
            
        # Update best
        if current_m > best_m:
            best_m = current_m
            best_adj = adj[:]
            
        if idx == len(all_edges):
            return
            
        # Bound
        ub = upper_bound(adj, current_m)
        if ub <= best_m:
            return
            
        u, v = all_edges[idx]
        
        # Option 1: Skip edge (u, v)
        dfs(idx + 1, adj, current_m)
        
        # Option 2: Add edge (u, v)
        # Check validity
        # Adding (u, v) increases common neighbors of (u, w) and (v, w) for all w.
        # Specifically, for any w, if w is neighbor of v, common(u, w) increases by 1.
        # If w is neighbor of u, common(v, w) increases by 1.
        
        # Check if adding (u, v) is valid
        valid = True
        
        # Check pair (u, v) itself: common(u, v) must be <= 2.
        # Current common(u, v) is bin(adj[u] & adj[v]).count('1')
        # Adding (u, v) does not change common(u, v) as derived before.
        # So check current common(u, v) <= 2.
        if bin(adj[u] & adj[v]).count('1') > 2:
            valid = False
            
        if valid:
            # Check effects on other pairs
            # For each w != u, v:
            # If w in N(v), then common(u, w) increases. Must be <= 1 before.
            # If w in N(u), then common(v, w) increases. Must be <= 1 before.
            
            # Iterate neighbors of v
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                mask_v ^= lsb
                if w == u:
                    continue
                # w is neighbor of v
                # Check common(u, w)
                if bin(adj[u] & adj[w]).count('1') > 1:
                    valid = False
                    break
            
            if valid:
                mask_u = adj[u]
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    mask_u ^= lsb
                    if w == v:
                        continue
                    # w is neighbor of u
                    # Check common(v, w)
                    if bin(adj[v] & adj[w]).count('1') > 1:
                        valid = False
                        break
        
        if valid:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            dfs(idx + 1, adj, current_m + 1)
            
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    # To speed up, we can order edges.
    # Edges involving high-degree potential vertices first?
    # Or just natural order.
    
    # Initial guess for best_m to help pruning?
    # Let's run a quick greedy to get a lower bound.
    def greedy_lower_bound():
        adj_g = [0] * n
        m_g = 0
        # Try adding edges in random order or degree order
        # Let's just do a simple greedy: try all edges, add if valid.
        edges_to_try = all_edges[:]
        # Randomize for variety? No, deterministic.
        # Sort by something?
        for u, v in edges_to_try:
            if time.time() > deadline - 10:
                break
            # Check validity
            valid = True
            if bin(adj_g[u] & adj_g[v]).count('1') > 2:
                valid = False
            if valid:
                mask_v = adj_g[v]
                while mask_v and valid:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    mask_v ^= lsb
                    if w != u and bin(adj_g[u] & adj_g[w]).count('1') > 1:
                        valid = False
            if valid:
                mask_u = adj_g[u]
                while mask_u and valid:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    mask_u ^= lsb
                    if w != v and bin(adj_g[v] & adj_g[w]).count('1') > 1:
                        valid = False
            
            if valid:
                adj_g[u] |= (1 << v)
                adj_g[v] |= (1 << u)
                m_g += 1
        return m_g, adj_g

    init_m, init_adj = greedy_lower_bound()
    best_m = init_m
    best_adj = init_adj
    
    # Run B&B
    # Note: B&B on 190 edges is huge.
    # The bound is critical.
    # The degree bound sum C(d,2) <= 380 is quite tight.
    
    dfs(0, [0]*n, 0)
    
    # Convert best_adj to edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    max_c = 0
    for i in range(20):
        for j in range(i + 1, 20):
            c = bin(adj[i] & adj[j]).count('1')
            if c > max_c:
                max_c = c
    print(f"Max common neighbors: {max_c}")
    if max_c <= 2:
        print("Valid")
    else:
        print("Invalid")
