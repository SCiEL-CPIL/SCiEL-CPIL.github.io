"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 15}

def _validate(edges, n):
    """Check if the graph satisfies the constraint.
    For every simple path s-u-v-t (3 edges), endpoints s and t must have different degrees in the full graph.
    """
    # Build adjacency and degree
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False  # duplicate
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Check all paths of length 3: s-u-v-t
    # A path is s-u-v-t where (s,u), (u,v), (v,t) are edges, all distinct vertices
    # For each middle edge (u,v), we need to check all pairs (s, t) where s in adj[u], t in adj[v], s != t
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue  # avoid double counting
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _edge_index(edge, n):
    """Convert edge (u,v) to an index in the complete graph edge list."""
    u, v = edge
    if u > v:
        u, v = v, u
    # Index in triangular number scheme
    # For n vertices, edge (i,j) with i<j has index sum_{k=0}^{i-1}(n-1-k) + (j-i-1)
    # Simpler: just use a dict
    return (u, v)

def _build_conflict_info(n):
    """
    For a given graph, compute which pairs of edges conflict.
    Two edges conflict if they can be part of a forbidden path.
    Actually, the constraint is global, so we need a different approach.
    
    Let me think differently: we want to find a maximum-size subset of edges such that
    no forbidden path exists. This is a constraint satisfaction / maximum independent set problem
    on a hypergraph (since constraints involve 3 edges).
    
    Alternative approach: use local search. Start with a random graph, then iteratively
    add/remove edges to satisfy constraints while maximizing edge count.
    """
    pass

def construct():
    n = P['n']
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Generate all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    total_edges = len(all_edges)
    
    # We'll use a local search approach:
    # 1. Start with a random subset of edges
    # 2. Try to add edges that don't create violations
    # 3. Try to swap edges to increase count
    # 4. Repeat until deadline
    
    def check_violation(edges_set, deg, adj, candidate_edge):
        """Check if adding candidate_edge to the graph creates any violation.
        Only check paths that involve the new edge."""
        u, v = candidate_edge
        # New edge is (u,v). Paths involving this edge as one of the 3 edges:
        # Case 1: new edge is the middle edge (u,v), path is s-u-v-t
        #   s in adj[u] \ {v}, t in adj[v] \ {u}, s != t
        #   Need deg[s] != deg[t]
        # Case 2: new edge is the first edge (s,u), path is s-u-v-t where (u,v) is existing
        #   For each w in adj[u] \ {v}: path s-u-w-t where t in adj[w] \ {u}, t != s
        #   Need deg[s] != deg[t]
        # Case 3: new edge is the last edge (v,t), path is s-u-v-t where (u,v) is existing
        #   For each w in adj[v] \ {u}: path s-w-v-t where s in adj[w] \ {v}, s != t
        #   Need deg[s] != deg[t]
        
        # Case 1: new edge is middle
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return True
        
        # Case 2: new edge is first (s=u is the endpoint, u-v is... wait)
        # Actually, the new edge is (u,v). In path s-a-b-t, the edges are (s,a), (a,b), (b,t).
        # The new edge could be any of these three.
        # Let me re-enumerate:
        # The new edge (u,v) can be:
        # - Edge 1: (s,a) = (u,v), so s=u, a=v. Then path is u-v-b-t where (v,b) and (b,t) are existing.
        # - Edge 2: (a,b) = (u,v), so a=u, b=v. Then path is s-u-v-t where (s,u) and (v,t) are existing.
        # - Edge 3: (b,t) = (u,v), so b=u, t=v. Then path is s-w-u-v where (s,w) and (w,u) are existing.
        
        # Case 2: new edge is edge 1: u-v-b-t
        for b in adj[v]:
            if b == u:
                continue
            for t in adj[b]:
                if t == v or t == u:
                    continue
                if deg[u] == deg[t]:
                    return True
        
        # Case 3: new edge is edge 3: s-w-u-v
        for w in adj[u]:
            if w == v:
                continue
            for s in adj[w]:
                if s == u or s == v:
                    continue
                if deg[s] == deg[v]:
                    return True
        
        return False
    
    def full_check(edges_list):
        """Full validation of a complete edge list."""
        return _validate(edges_list, n)
    
    best_edges = None
    best_count = 0
    
    # Strategy: greedy construction with multiple random orderings
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        random.shuffle(all_edges)
        
        edges_set = set()
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        for e in all_edges:
            if time.time() > deadline - 5:
                break
            if check_violation(edges_set, deg, adj, e):
                continue
            # Add edge
            u, v = e
            edges_set.add(e)
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        edge_count = len(edges_set)
        if edge_count > best_count:
            # Verify
            if full_check(list(edges_set)):
                best_count = edge_count
                best_edges = list(edges_set)
        
        # Local improvement: try to add edges that were skipped
        # Try swapping: remove one edge, try to add two
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges_list = list(edges_set)
            for i in range(len(edges_list)):
                if time.time() > deadline - 5:
                    break
                # Try removing edge i
                e_remove = edges_list[i]
                u_r, v_r = e_remove
                # Remove
                adj[u_r].discard(v_r)
                adj[v_r].discard(u_r)
                deg[u_r] -= 1
                deg[v_r] -= 1
                edges_set.discard(e_remove)
                
                # Try adding other edges
                added_any = False
                for e_add in all_edges:
                    if e_add in edges_set:
                        continue
                    if not check_violation(edges_set, deg, adj, e_add):
                        u_a, v_a = e_add
                        edges_set.add(e_add)
                        adj[u_a].add(v_a)
                        adj[v_a].add(u_a)
                        deg[u_a] += 1
                        deg[v_a] += 1
                        added_any = True
                        break  # Only add one for now
                
                if not added_any:
                    # Put back
                    edges_set.add(e_remove)
                    adj[u_r].add(v_r)
                    adj[v_r].add(u_r)
                    deg[u_r] += 1
                    deg[v_r] += 1
                else:
                    improved = True
                    break
        
        # Update best
        edge_count = len(edges_set)
        if edge_count > best_count:
            if full_check(list(edges_set)):
                best_count = edge_count
                best_edges = list(edges_set)
        
        if best_count >= total_edges:
            break
    
    if best_edges is None:
        # Fallback: star graph
        best_edges = [[0, v] for v in range(1, n)]
    
    # Convert to proper format
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
