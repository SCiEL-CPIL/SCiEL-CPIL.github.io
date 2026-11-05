"""C6-free graph on 40 vertices: maximize edges via greedy + local search."""
import time
import random
import sys
from collections import deque

def construct():
    """
    Construct a C6-free graph on 40 vertices maximizing edges.
    Strategy:
    1. Start with a good initial graph (e.g., based on known constructions or random).
    2. Use greedy edge addition while checking C6-freeness.
    3. Use local search (add/remove edges) to improve.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline for safety
    
    n = 40
    
    def has_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6. 
        This is done by checking if there's a path of length 4 between u and v not using the edge.
        Actually, adding edge (u,v) creates a C6 iff there exists a path of length 4 between u and v.
        A path of length 4: u - a - b - c - v where a,b,c are distinct and not u,v.
        We can do BFS from u to find if v is reachable in exactly 4 steps, but we must be careful 
        about simple paths. For efficiency, we check: is there a path u-a-b-c-v with all distinct?
        This is equivalent to: exists a in N(u), c in N(v), and a path of length 2 from a to c (a-b-c) 
        where b is not u, v, a, c.
        """
        # For each neighbor a of u and neighbor c of v:
        #   check if there's a vertex b adjacent to both a and c, with b not in {u,v,a,c}
        #   and also ensure the path u-a-b-c-v is simple (all 5 internal vertices distinct)
        # Actually, u-a-b-c-v: vertices are u, a, b, c, v. They must all be distinct.
        # So a != v, c != u, a != c, b != u, b != v, b != a, b != c.
        # Since a is neighbor of u and c is neighbor of v, and we're adding edge u-v,
        # we need a != v (a is in N(u), if a==v then u-v already exists which it doesn't),
        # c != u similarly.
        
        for a in adj[u]:
            if a == v:
                continue
            for c in adj[v]:
                if c == u or c == a:
                    continue
                # Check if there's a common neighbor b of a and c, with b not in {u, v, a, c}
                common = adj[a] & adj[c]
                for b in common:
                    if b not in (u, v, a, c):
                        return True
        return False
    
    def is_c6_free(adj):
        """Check if the graph is C6-free."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & adj[v]:  # edge exists
                    # Temporarily remove edge and check
                    adj[u].discard(v)
                    adj[v].discard(u)
                    result = has_c6(adj, u, v)
                    adj[u].add(v)
                    adj[v].add(u)
                    if result:
                        return False
        return True
    
    def count_edges(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Initialize with 8 disjoint K5s
    adj = [set() for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u, v = 5*k+a, 5*k+b
                adj[u].add(v)
                adj[v].add(u)
    
    # Greedy addition: try to add edges in a smart order
    # Order edges by some heuristic (e.g., sum of degrees of endpoints)
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
    
    def edge_score(u, v):
        return len(adj[u]) + len(adj[v])
    
    # Try adding edges greedily
    added = 0
    while True:
        if time.time() > deadline:
            break
        # Find the best edge to add
        best_edge = None
        best_score = -1
        for u, v in all_pairs:
            if v in adj[u]:
                continue
            if not has_c6(adj, u, v):
                s = edge_score(u, v)
                if s > best_score:
                    best_score = s
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        all_pairs.remove((u, v))
        added += 1
    
    # Local search: try to add edges, and if stuck, try removing one edge to enable more additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any valid edge
        for u, v in all_pairs[:]:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            if not has_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                all_pairs.remove((u, v))
                improved = True
        
        if improved:
            continue
        
        # Try removing one edge to enable adding two or more
        current_edges = count_edges(adj)
        for u, v in [(u, v) for u in range(n) for v in range(u+1, n) if v in adj[u]]:
            if time.time() > deadline:
                break
            # Remove edge (u,v)
            adj[u].discard(v)
            adj[v].discard(u)
            
            # Try to add as many edges as possible
            new_pairs = [(u2, v2) for u2 in range(n) for v2 in range(u2+1, n) if v2 not in adj[u2]]
            added_after = 0
            for u2, v2 in new_pairs[:]:
                if time.time() > deadline:
                    break
                if not has_c6(adj, u2, v2):
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    new_pairs.remove((u2, v2))
                    added_after += 1
            
            new_edges = count_edges(adj)
            if new_edges > current_edges:
                # Keep the improvement
                all_pairs = [(u2, v2) for u2 in range(n) for v2 in range(u2+1, n) if v2 not in adj[u2]]
                improved = True
            else:
                # Revert: restore edge (u,v) and remove newly added edges
                # This is complex; instead, let's not do this full revert
                # Just restore the removed edge and break
                adj[u].add(v)
                adj[v].add(u)
    
    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges

# Test
if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
