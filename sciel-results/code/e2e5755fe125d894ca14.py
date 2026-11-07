"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 15}

def _all_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _degrees_from_edges(edges, n):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg

def _is_valid(edges, n):
    """Check if the graph has no 3-edge path with equal-degree endpoints."""
    deg = _degrees_from_edges(edges, n)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each edge (u,v), for each s in adj[u] (s != v), for each t in adj[v] (t != u):
    # path s-u-v-t is a 3-edge path if s != t
    # We need deg[s] != deg[t]
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _construct_greedy(n, max_time=85.0):
    """
    Greedy + local search to maximize edges.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    all_edges = _all_edges(n)
    m = len(all_edges)
    
    # Start with a random subset, then greedily add edges
    # Try multiple random orderings and keep the best
    
    best_edges = []
    best_count = 0
    
    num_tries = 20
    for trial in range(num_tries):
        if time.time() > deadline - 5.0:
            break
        
        # Random shuffle of edges
        edges_order = list(all_edges)
        random.shuffle(edges_order)
        
        current_edges = []
        deg = [0] * n
        adj = [set() for _ in range(n)]
        
        # Greedily add edges in random order
        for e in edges_order:
            if time.time() > deadline - 2.0:
                break
            u, v = e
            # Tentatively add
            # Check if adding this edge creates a forbidden path
            # A forbidden path s-u-v-t requires:
            # - s is neighbor of u (s != v)
            # - t is neighbor of v (t != u)  
            # - s != t
            # - deg[s] == deg[t] (in the FULL graph including this new edge)
            
            # When we add edge (u,v), deg[u] and deg[v] each increase by 1
            # For a path s-u-v-t to be forbidden after adding:
            #   deg_new[s] == deg_new[t]
            #   where deg_new[x] = deg[x] + (1 if x==u or x==v else 0)
            
            valid = True
            for s in adj[u]:
                if s == v:
                    continue
                deg_s = deg[s] + (1 if s == u else 0)  # s != u since s is in adj[u] and no self-loops, but s could be... wait s is neighbor of u, s != u always. Actually s is in adj[u], so s != u. But s could equal v? We skip s==v.
                # Actually s is a neighbor of u, so s != u. s could be v but we skip that.
                # deg_s = deg[s] (since s != u and s != v in the loop body when s != v)
                # Wait, s is in adj[u], so s is a neighbor. s != u. If s != v (we skip v), then s is neither u nor v.
                # So deg_new[s] = deg[s]
                
                for t in adj[v]:
                    if t == u:
                        continue
                    if t == s:
                        continue
                    # t is in adj[v], t != u. If t != s (checked), t could be u? No, we skip t==u.
                    # t could be v? No, t is in adj[v] so t != v.
                    # t could be u? Skipped.
                    # So t is neither u nor v (since t != u and t != v).
                    # deg_new[t] = deg[t]
                    
                    # But wait: we need to check if deg[s] == deg[t] in the NEW graph.
                    # Since s != u,v and t != u,v, deg_new[s] = deg[s] and deg_new[t] = deg[t].
                    if deg[s] == deg[t]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_edges.append(e)
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
        
        # Local search: try to add more edges
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try each edge not in current_edges
            current_set = set(tuple(e) for e in current_edges)
            for e in all_edges:
                if time.time() > deadline - 2.0:
                    break
                if tuple(e) in current_set:
                    continue
                u, v = e
                # Check if we can add
                valid = True
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if t == s:
                            continue
                        # deg_new[s] and deg_new[t]
                        # s is neighbor of u, s != v, so s != u, s != v
                        # t is neighbor of v, t != u, t != s, so t != u, t != v
                        if deg[s] == deg[t]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current_edges.append(e)
                    current_set.add(tuple(e))
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
                    improved = True
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
    
    # Verify the best result
    if best_edges and _is_valid(best_edges, n):
        return best_edges
    else:
        # Fallback: return empty or minimal valid
        return []

def construct():
    n = P['n']
    edges = _construct_greedy(n, max_time=85.0)
    # Ensure types are correct
    result = [[int(u), int(v)] for u, v in edges]
    return result
# EVOLVE-BLOCK-END
