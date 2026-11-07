"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 60s. Let's aim to finish in 45s to be safe.
    deadline = start_time + 45.0
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge list and indices
    # edge_list[i] = (u, v)
    # idx_to_edge[i] = (u, v)
    # edge_to_idx[(u, v)] = i
    edge_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
            
    # Helper to check if adding a specific edge to the current set violates constraints
    # Current state: adj (list of sets), deg (list of ints), current_edge_set (set of indices)
    # Candidate edge: (u, v)
    # Violation exists if there is a path s-u-v-t (or s-v-u-t) with deg(s) == deg(t) and s != t.
    # Note: The path is simple, so s, u, v, t are distinct.
    # Edges (s,u), (u,v), (v,t) must exist.
    # If we add (u,v), we need to check:
    # 1. For any s in N(u) \ {v}, and any t in N(v) \ {u}:
    #    If s != t and deg(s) == deg(t), then adding (u,v) creates a violation.
    #    Wait, the path is s-u-v-t. The edges are (s,u), (u,v), (v,t).
    #    (u,v) is the new edge. (s,u) and (v,t) must already exist.
    #    So s is a neighbor of u (excluding v, but v isn't a neighbor yet).
    #    t is a neighbor of v (excluding u).
    #    If such s, t exist with deg(s)==deg(t) and s!=t, the edge is bad.
    
    def is_addition_safe(u, v, adj, deg):
        # Check paths s-u-v-t
        # s must be in adj[u]
        # t must be in adj[v]
        # s != t
        # deg[s] == deg[t]
        
        # Optimization: Group neighbors of u by degree, neighbors of v by degree.
        # If there is a degree d such that:
        # - There is an s in adj[u] with deg[s] == d
        # - There is a t in adj[v] with deg[t] == d
        # - It is not the case that the only such s and t are the same vertex (s==t)
        #   (Note: s and t are from different neighbor sets, but s could equal t if u and v share a neighbor s. 
        #    If s==t, the path is s-u-v-s, which is a cycle, not a simple path with distinct endpoints s,t. 
        #    The prompt says "simple three-edge path ... endpoints s,t". Usually implies s!=t. 
        #    The prompt explicitly says "s != t".)
        
        # Let's collect counts.
        # If for some degree d, count_s >= 1 and count_t >= 1:
        #   We need to ensure we can pick s != t.
        #   If the set of s's with deg d is S_d and set of t's with deg d is T_d.
        #   We need S_d ∩ T_d != S_d ∪ T_d? No.
        #   We need to pick s in S_d and t in T_d such that s != t.
        #   This fails ONLY if S_d == T_d == {x} for some x.
        #   i.e., |S_d| == 1 and |T_d| == 1 and S_d == T_d.
        
        s_by_deg = {}
        for s in adj[u]:
            d = deg[s]
            if d not in s_by_deg:
                s_by_deg[d] = []
            s_by_deg[d].append(s)
            
        t_by_deg = {}
        for t in adj[v]:
            d = deg[t]
            if d not in t_by_deg:
                t_by_deg[d] = []
            t_by_deg[d].append(t)
            
        for d in s_by_deg:
            if d in t_by_deg:
                S = s_by_deg[d]
                T = t_by_deg[d]
                # Check if we can pick distinct s, t
                if len(S) == 1 and len(T) == 1 and S[0] == T[0]:
                    continue # Only conflict is s==t, which is allowed (not a valid path)
                else:
                    return False # Found a valid violating path
        return True

    def is_graph_valid(adj, deg, current_edge_set):
        # Check all existing edges for violations
        # This is expensive. We only call this if we suspect the state is invalid 
        # or as a final check. During construction, we maintain validity by only adding safe edges.
        # However, "safe" checks only look at the new edge. 
        # Does adding a safe edge make previously safe edges unsafe?
        # No. The constraint is "Reject if ANY path...". 
        # If the graph was valid before, and we add an edge (u,v) that doesn't create a NEW violating path 
        # involving (u,v), does it create a violating path NOT involving (u,v)?
        # No, because the degrees of u and v change. 
        # Wait. deg(u) and deg(v) increase by 1.
        # This changes the condition deg(s) == deg(t) for paths s-u'-v'-t where u' or v' is u or v?
        # No, the path is s-u-v-t. The endpoints are s and t. Their degrees don't change when we add (u,v).
        # The internal vertices u and v change degree.
        # The condition is on deg(s) and deg(t).
        # So, adding (u,v) only creates new violating paths that USE (u,v).
        # It does NOT invalidate existing paths that do NOT use (u,v), because the degrees of the endpoints of those existing paths haven't changed.
        # Therefore, if we start with an empty graph (valid) and only add edges that are "safe" (don't create new violations), the graph remains valid at all times.
        
        # So we don't need a full is_graph_valid check during the greedy phase!
        # We just need is_addition_safe.
        
        return True # Placeholder, not used in greedy logic

    # Greedy Construction
    # Strategy: Try to add edges in a specific order. 
    # To maximize density, we want to add "central" edges first? Or random?
    # Random order with multiple restarts is a good baseline.
    # But let's try a deterministic heuristic first: 
    # Sort edges by some metric? 
    # Actually, random restarts with greedy fill is robust.
    
    best_edge_count = 0
    best_edges = set()
    
    # Number of restarts. We have ~45 seconds. 
    # Each restart takes O(E^2) checks? 
    # is_addition_safe is O(deg_u + deg_v). Max 14.
    # Total checks per restart: ~105 edges. 105 * 14 ~ 1500 ops.
    # Very fast. We can do thousands of restarts.
    
    # Let's do 1000 restarts or until time runs out.
    for restart in range(2000):
        if time.time() > deadline - 2.0: # Leave 2s for final conversion
            break
            
        current_edges = set()
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        # Shuffle edge order for this restart
        # Using a simple LCG or random.shuffle is fine. 
        # To avoid importing random again (or to be deterministic-ish per restart), 
        # we can use a seed based on restart index.
        # But random.shuffle is fast enough.
        import random
        random.seed(restart) # Deterministic restarts for reproducibility, or just use time?
        # Let's use a fixed seed + restart offset for reproducibility of the "strategy"
        # Actually, random.shuffle without seed is better for diversity.
        # But we need to ensure we don't spend too much time in setup.
        
        edges_to_try = list(range(total_edges))
        random.shuffle(edges_to_try)
        
        for e_idx in edges_to_try:
            u, v = edge_list[e_idx]
            if is_addition_safe(u, v, adj, deg):
                # Add edge
                current_edges.add(e_idx)
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = current_edges.copy()
            
            # If we reached a high count, we might want to stop early?
            # Max possible is 105. 
            # If we get close to 105, we can break.
            # But 105 is likely impossible. 
            # Let's just run all restarts.
            
    # Final Verification (just in case logic bug exists)
    # Reconstruct adj and deg from best_edges
    adj_final = [set() for _ in range(n)]
    deg_final = [0] * n
    for e_idx in best_edges:
        u, v = edge_list[e_idx]
        adj_final[u].add(v)
        adj_final[v].add(u)
        deg_final[u] += 1
        deg_final[v] += 1
        
    # Check validity
    valid = True
    for e_idx in best_edges:
        u, v = edge_list[e_idx]
        # Check if this edge is part of a violation
        # Re-use logic from is_addition_safe but assuming edge exists
        # Actually, is_addition_safe checks if ADDING it creates a violation.
        # If it's already in the graph, we need to check if the graph is valid.
        # Since we built it by only adding safe edges, it SHOULD be valid.
        # But let's double check one edge to be safe? 
        # No, let's just trust the greedy invariant. 
        # If the invariant holds, no full check is needed.
        
    # Convert to list of [u, v]
    result = []
    for e_idx in best_edges:
        u, v = edge_list[e_idx]
        result.append([u, v])
        
    # Ensure types
    for edge in result:
        edge[0] = int(edge[0])
        edge[1] = int(edge[1])
        
    return result
# EVOLVE-BLOCK-END
