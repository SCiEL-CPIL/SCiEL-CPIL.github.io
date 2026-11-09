import time
import random
import math
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

def construct():
    """
    Maximize edges in a K2,3-free graph on 20 vertices.
    
    Strategy:
    1. Attempt exact optimization using MILP (pulp/CBC) if available.
    2. If MILP is unavailable or times out, use high-quality heuristic:
       - Fast Greedy with random restarts.
       - Local Search with 1-edge swaps and 2-edge swaps.
    3. Return the best valid solution found within the time limit.
    """
    n = 20
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                temp -= lsb
        return edges

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_k23_free(adj):
        """
        Check if the graph is K2,3-free.
        Condition: For all u, v, |N(u) & N(v)| <= 2.
        """
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    def can_add_edge(adj, u, v):
        """
        Fast check if adding edge (u,v) keeps the graph K2,3-free.
        1. |N(u) & N(v)| <= 2
        2. For w in N(u)\{v}, |N(w) & N(v)| <= 1 (since u joins N(w) & N(v))
        3. For w in N(v)\{u}, |N(w) & N(u)| <= 1 (since v joins N(w) & N(u))
        """
        # 1. Common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if bin(common_uv).count('1') > 2:
            return False
        
        # 2. Check neighbors of u
        # u is already in adj[v]? No, we are adding it.
        # So for any w in N(u), u is a common neighbor of w and v? No.
        # If we add (u,v), then u is in N(v) and v is in N(u).
        # For any w in N(u), w is a common neighbor of u and ? No.
        # The condition is about pairs of vertices having 3 common neighbors.
        # If we add (u,v), the common neighborhood sizes change for:
        # - Pair (u,v): was |N(u)&N(v)|, now +1? No, N(u) and N(v) don't change relative to each other? 
        #   Wait. N(u) gains v. N(v) gains u.
        #   Common neighbors of u and v are vertices x such that x in N(u) and x in N(v).
        #   Adding edge (u,v) does NOT change N(u) \ {v} or N(v) \ {u}.
        #   So |N(u) & N(v)| remains the same? 
        #   NO. N(u) is the set of neighbors. If we add edge (u,v), v enters N(u) and u enters N(v).
        #   Is v in N(u) & N(v)? No, v is not in N(v) (simple graph).
        #   Is u in N(u) & N(v)? No.
        #   So the set of common neighbors of u and v is exactly the set of vertices w != u,v such that w is adjacent to both.
        #   This set does NOT change when we add edge (u,v).
        #   So Condition 1 is: |N(u) & N(v)| <= 2. This is a static check on the current graph.
        
        # What about other pairs?
        # Consider pair (u, w) where w in N(u).
        # Currently, common neighbors of u and w are N(u) & N(w).
        # If we add (u,v), does v become a common neighbor of u and w?
        # v is in N(u) (new). Is v in N(w)?
        # If v in N(w), then v is a new common neighbor of u and w.
        # So for every w in N(u), if v in N(w), then |N(u) & N(w)| increases by 1.
        # We require |N(u) & N(w)| <= 2.
        # So we need: for all w in N(u) such that v in N(w), current |N(u) & N(w)| must be <= 1.
        # Similarly for pair (v, w) where w in N(v).
        # If u in N(w), then u is a new common neighbor of v and w.
        # So for all w in N(v) such that u in N(w), current |N(v) & N(w)| must be <= 1.
        
        # Let's verify this logic.
        # Current graph G. Add edge (u,v) to get G'.
        # For any pair (a,b) in G':
        # If {a,b} = {u,v}, CN(a,b) in G' = CN(a,b) in G (since u,v not in their own neighborhoods).
        # If a=u, b=w (w in N(u)).
        #   CN(u,w) in G' = CN(u,w) in G U {v} if v in N(w) and v in N(u).
        #   v is in N(u) by definition of w in N(u)? No, w in N(u) means (u,w) edge.
        #   v is in N(u) because we added (u,v).
        #   So v is in N(u).
        #   If v is also in N(w), then v is in N(u) & N(w).
        #   So CN(u,w) increases by 1 if (w,v) is an edge.
        #   We need CN(u,w) <= 2. So if (w,v) edge, we need current CN(u,w) <= 1.
        # If a=w, b=v (w in N(v)).
        #   CN(w,v) in G' = CN(w,v) in G U {u} if u in N(w).
        #   u is in N(v) (added edge).
        #   If u is in N(w), then u is in N(w) & N(v).
        #   So CN(w,v) increases by 1 if (w,u) is an edge.
        #   We need CN(w,v) <= 2. So if (w,u) edge, we need current CN(w,v) <= 1.
        # If {a,b} is disjoint from {u,v}, CN unchanged.
        
        # So the checks are:
        # 1. bin(adj[u] & adj[v]).count('1') <= 2
        # 2. For all w in N(u) such that (w,v) is edge: bin(adj[u] & adj[w]).count('1') <= 1
        # 3. For all w in N(v) such that (w,u) is edge: bin(adj[v] & adj[w]).count('1') <= 1
        
        # Check 1
        if bin(adj[u] & adj[v]).count('1') > 2:
            return False
            
        # Check 2: w in N(u) AND v in N(w)  <=>  w in N(u) & N(v)
        # Wait, v in N(w) is equivalent to w in N(v).
        # So w is in N(u) & N(v).
        # So for all w in N(u) & N(v):
        #   We need current CN(u,w) <= 1.
        #   CN(u,w) is bin(adj[u] & adj[w]).count('1').
        common_uv = adj[u] & adj[v]
        temp = common_uv
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if bin(adj[u] & adj[w]).count('1') > 1:
                return False
            if bin(adj[v] & adj[w]).count('1') > 1:
                return False
        
        return True

    def greedy_fill(adj, order=None, deadline=None):
        if order is None:
            order = all_pairs[:]
        for u, v in order:
            if deadline is not None and time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)

    def local_search(adj, deadline):
        """
        Local search with 1-edge swaps.
        Try to remove a random edge e1 and add a random non-edge e2.
        If valid, accept.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        
        # Precompute non-edges for faster sampling? 
        # With n=20, 190 edges total. Non-edges ~ 90. 
        # Random sampling is fine.
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick a random edge to remove
            u1, v1 = random.choice(all_pairs)
            if not (adj[u1] & (1 << v1)):
                continue # Not an edge
            
            # Pick a random non-edge to add
            u2, v2 = random.choice(all_pairs)
            if adj[u2] & (1 << v2):
                continue # Already an edge
            
            # Perform swap
            remove_edge(adj, u1, v1)
            if can_add_edge(adj, u2, v2):
                add_edge(adj, u2, v2)
                current_edges = count_edges(adj)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Revert
                add_edge(adj, u1, v1)
                no_improve_count += 1
                
            # Occasionally do a 2-move? 
            # For now, stick to 1-move to keep it simple and fast.
            
        return best_adj, best_edges

    def local_search_2move(adj, deadline):
        """
        Local search with 2-edge swaps.
        Remove e1, e2 and add e3, e4.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        no_improve_count = 0
        max_no_improve = 200 # 2-moves are more expensive
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick 2 edges to remove
            e1 = random.choice(all_pairs)
            e2 = random.choice(all_pairs)
            if e1[0] == e2[0] and e1[1] == e2[1]:
                continue
            if not (adj[e1[0]] & (1 << e1[1])) or not (adj[e2[0]] & (1 << e2[1])):
                continue
            
            # Pick 2 non-edges to add
            e3 = random.choice(all_pairs)
            e4 = random.choice(all_pairs)
            if e3[0] == e4[0] and e3[1] == e4[1]:
                continue
            # e3, e4 must be distinct from each other
            if (adj[e3[0]] & (1 << e3[1])) or (adj[e4[0]] & (1 << e4[1])):
                continue
            # e3, e4 should not be the same as e1, e2? 
            # If e3 == e1, then we are just removing e2 and adding e4? 
            # It's safer to ensure they are distinct non-edges in the graph AFTER removal.
            # But simpler: just try.
            
            u1, v1 = e1
            u2, v2 = e2
            u3, v3 = e3
            u4, v4 = e4
            
            # Remove e1, e2
            remove_edge(adj, u1, v1)
            remove_edge(adj, u2, v2)
            
            # Try to add e3
            if can_add_edge(adj, u3, v3):
                add_edge(adj, u3, v3)
                # Try to add e4
                if can_add_edge(adj, u4, v4):
                    add_edge(adj, u4, v4)
                    current_edges = count_edges(adj)
                    no_improve_count = 0
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Revert e3
                    remove_edge(adj, u3, v3)
                    # Revert e1, e2
                    add_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)
                    no_improve_count += 1
            else:
                # Revert e1, e2
                add_edge(adj, u1, v1)
                add_edge(adj, u2, v2)
                no_improve_count += 1
                
        return best_adj, best_edges

    best_adj = None
    best_edges = 0

    # --- Strategy 1: Try ILP ---
    try:
        import pulp
        import itertools
        
        # Time budget for ILP
        ilp_time_limit = 30
        
        prob = pulp.LpProblem("K23_free_max_edges", pulp.LpMaximize)
        x = {}
        for u in range(n):
            for v in range(u + 1, n):
                x[u, v] = pulp.LpVariable(f"x_{u}_{v}", cat='Binary')
        
        prob += pulp.lpSum(x[u, v] for u in range(n) for v in range(u + 1, n))
        
        # Constraints: for all disjoint U={u1,u2}, V={v1,v2,v3}, sum(x) <= 5
        for U in itertools.combinations(range(n), 2):
            for V in itertools.combinations(range(n), 3):
                if len(set(U) & set(V)) > 0:
                    continue
                edge_sum = 0
                for u in U:
                    for v in V:
                        if u < v:
                            edge_sum += x[u, v]
                        else:
                            edge_sum += x[v, u]
                prob += edge_sum <= 5
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=ilp_time_limit))
        
        if pulp.LpStatus[prob.status] == "Optimal":
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if x[u, v].value() is not None and x[u, v].value() > 0.5:
                        add_edge(adj, u, v)
            edges_found = count_edges(adj)
            if is_k23_free(adj):
                if edges_found > best_edges:
                    best_edges = edges_found
                    best_adj = adj[:]
        
    except Exception as e:
        pass

    # --- Strategy 2: Heuristic ---
    deadline = start_time + time_limit
    
    # If ILP didn't find anything, or we have time, run heuristics
    # If ILP found a good solution, we can use it as a starting point for local search
    
    if best_adj is None:
        # Run multiple random restarts
        num_restarts = 50
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            adj = [0] * n
            order = all_pairs[:]
            random.shuffle(order)
            greedy_fill(adj, order, deadline)
            
            curr_edges = count_edges(adj)
            if curr_edges > best_edges:
                best_edges = curr_edges
                best_adj = adj[:]
                
    else:
        # Use ILP result as seed for local search
        pass

    # Local Search Phase 1: 1-moves
    if best_adj is not None and time.time() < deadline - 5:
        improved_adj, improved_edges = local_search(best_adj[:], deadline - 5)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj = improved_adj

    # Local Search Phase 2: 2-moves
    if best_adj is not None and time.time() < deadline - 2:
        improved_adj, improved_edges = local_search_2move(best_adj[:], deadline - 2)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj = improved_adj

    # Final Greedy Pass
    if best_adj is not None and time.time() < deadline:
        # Try to add any remaining edges
        greedy_fill(best_adj, deadline=deadline)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    if best_adj is None:
        # Fallback: empty graph or simple construction
        best_adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if can_add_edge(best_adj, u, v):
                    add_edge(best_adj, u, v)

    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
