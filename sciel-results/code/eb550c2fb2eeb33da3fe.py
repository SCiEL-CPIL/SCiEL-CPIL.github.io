import time
import itertools
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    n = 15
    A = list(range(7))
    B = list(range(7, 15))
    
    # Start with complete bipartite K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((min(a,b), max(a,b)))
    
    candidates = []
    for i, j in itertools.combinations(A, 2):
        candidates.append((i, j))
    for i, j in itertools.combinations(B, 2):
        candidates.append((i, j))
    
    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid_incremental(current_edges, deg, adj, new_edge):
        """
        Check if adding new_edge to current_edges (with given deg, adj) creates a violation.
        Only checks paths involving new_edge.
        """
        u, v = new_edge
        # New paths are s-u-v-t and s-v-u-t
        # s is neighbor of u (excluding v), t is neighbor of v (excluding u)
        # deg[s] == deg[t] ?
        
        # Note: deg and adj are for the graph BEFORE adding new_edge.
        # Adding new_edge increases deg[u] and deg[v] by 1.
        # It does NOT change deg[s] or deg[t] for s != u, v and t != u, v.
        # So we can use the old deg for s and t.
        
        # However, if s == v or t == u, we skip.
        # If s == u or t == v, we skip.
        
        # Case 1: Path s-u-v-t
        # s in adj[u], s != v
        # t in adj[v], t != u
        # s != t
        # Check deg[s] == deg[t]
        
        # But wait, what if s or t is u or v?
        # s is neighbor of u. s != v (checked). s can be u? No, no loops.
        # t is neighbor of v. t != u (checked). t can be v? No.
        
        # So s and t are distinct from u, v.
        # Their degrees are unchanged by adding (u,v).
        
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        
        # Case 2: Path s-v-u-t
        # s in adj[v], s != u
        # t in adj[u], t != v
        # s != t
        for s in adj[v]:
            if s == u:
                continue
            for t in adj[u]:
                if t == v or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
                    
        return True

    def is_valid_full(edge_set):
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    best_edges = set(base_edges)
    best_count = len(base_edges)
    
    # Initial degrees and adjacency
    best_deg = get_degrees(best_edges)
    best_adj = [set() for _ in range(n)]
    for u, v in best_edges:
        best_adj[u].add(v)
        best_adj[v].add(u)
        
    for seed in range(100):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed)
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        current = set(base_edges)
        deg = get_degrees(current)
        adj = [set() for _ in range(n)]
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
            
        for i, j in cand_list:
            if time.time() > DEADLINE:
                break
            e = (min(i, j), max(i, j))
            if e in current:
                continue
            
            # Check incremental validity
            if is_valid_incremental(current, deg, adj, e):
                # Add edge
                u, v = e
                current.add(e)
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current) > best_count:
            best_edges = set(current)
            best_count = len(current)
            # Update best_deg and best_adj for potential use in next iteration? 
            # We restart from base, so not needed.
            
    # Local search on best_edges
    if time.time() < DEADLINE:
        current = set(best_edges)
        deg = get_degrees(current)
        adj = [set() for _ in range(n)]
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
            
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for e in list(current):
                if time.time() > DEADLINE:
                    break
                u_rem, v_rem = e
                # Remove e
                current.discard(e)
                deg[u_rem] -= 1
                deg[v_rem] -= 1
                adj[u_rem].discard(v_rem)
                adj[v_rem].discard(u_rem)
                
                # Try to add any candidate
                for ce in candidates:
                    if time.time() > DEADLINE:
                        break
                    if ce in current:
                        continue
                    
                    if is_valid_incremental(current, deg, adj, ce):
                        current.add(ce)
                        u_add, v_add = ce
                        deg[u_add] += 1
                        deg[v_add] += 1
                        adj[u_add].add(v_add)
                        adj[v_add].add(u_add)
                        improved = True
                        break # Restart loop? Or continue? 
                        # Breaking inner loop and continuing outer is safer for consistency.
                        
                # If we didn't add anything, restore e?
                # If we added something, we keep the removal of e and addition of ce.
                # If we didn't add anything, we should restore e to try other removals.
                if not improved:
                    current.add(e)
                    deg[u_rem] += 1
                    deg[v_rem] += 1
                    adj[u_rem].add(v_rem)
                    adj[v_rem].add(u_rem)
                    
                if len(current) > best_count:
                    best_edges = set(current)
                    best_count = len(current)
                    # Update deg/adj to match best? 
                    # We are modifying current. If we improve, we keep it.
                    # If we don't, we revert.
                    
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
