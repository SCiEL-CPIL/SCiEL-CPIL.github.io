import time
import random
import sys

def has_c6_with_edge(adj, u, v):
    """
    Check if adding edge (u, v) to the graph represented by adjacency lists 'adj'
    creates a C6.
    Assumes (u, v) is not currently in adj.
    """
    if u == v:
        return False
    
    # Get neighbors of u and v
    nu = adj[u]
    nv = adj[v]
    
    # We look for a path u-a-b-c-d-v of length 4.
    # a in nu, d in nv.
    # Path: a-b-c-d
    # b is neighbor of a.
    # c is neighbor of b.
    # d is neighbor of c.
    
    # Let's find all vertices 'c' that can be the 3rd node from u (distance 2 from u, via a,b)
    # And all vertices 'c' that can be the 3rd node from v (distance 2 from v, via d,b)
    # Wait, the path is u-a-b-c-d-v.
    # Distance from u to c is 2 (u-a-b-c? No, u-a is 1, a-b is 2, b-c is 3).
    # So c is at distance 2 from a? No.
    # u (0) -> a (1) -> b (2) -> c (3) -> d (4) -> v (5).
    # So we need a path of length 4 between u and v.
    # This is equivalent to a path of length 2 between N(u) and N(v).
    # Let A = N(u), B = N(v).
    # We need a path a-b-c-d where a in A, d in B.
    # This means b is a neighbor of some a in A.
    # And c is a neighbor of some d in B.
    # And b is a neighbor of c.
    
    # Step 1: Find all potential 'b's reachable from A in 1 step.
    # Actually, let's find the set of vertices reachable from A in 2 steps?
    # No, let's find the set of vertices 'c' that are distance 2 from A.
    # dist(A, c) = 2 means there exists a in A, b in N(a) such that c in N(b).
    # Let C_A = { c | exists a in A, exists b in N(a) such that (b,c) in E }
    # Similarly, let C_B = { c | exists d in B, exists c' in N(d) such that (c',c) in E }
    # Wait, the path is a-b-c-d.
    # b is connected to a.
    # c is connected to b.
    # d is connected to c.
    
    # So, let's compute the set of vertices reachable from A in 2 steps (let's call it Reach2_A).
    # Reach2_A = { w | exists a in A, exists b in N(a) such that (b,w) in E }
    # And Reach2_B = { w | exists d in B, exists c in N(d) such that (c,w) in E }
    
    # If Reach2_A and Reach2_B share a common neighbor?
    # No, if w is in Reach2_A, there is a path a-b-w.
    # If w is in Reach2_B, there is a path d-c-w.
    # This doesn't guarantee a path a-b-c-d.
    
    # Let's go back to:
    # We need a-b-c-d with a in A, d in B.
    # This is equivalent to:
    # There exists b such that b is neighbor of some a in A.
    # AND there exists c such that c is neighbor of some d in B.
    # AND b is neighbor of c.
    
    # So:
    # 1. Let N_A_1 = { b | exists a in A, (a,b) in E }  # Neighbors of A
    # 2. Let N_B_1 = { c | exists d in B, (d,c) in E }  # Neighbors of B
    # 3. Check if there is an edge (b,c) such that b in N_A_1 and c in N_B_1.
    #    AND b != a, c != d, b != c (simple path constraints).
    #    Also need to ensure the path doesn't reuse vertices.
    #    u, a, b, c, d, v.
    #    a in N(u), d in N(v).
    #    b in N(a), c in N(d).
    #    (b,c) in E.
    
    # Constraints:
    # a != b (guaranteed if simple graph and no self loops, but a is in N(u), b in N(a))
    # b != c (if b==c, path is u-a-b-d-v, length 4? No, u-a-b-d-v is length 4 edges. 
    #          Wait. u-a (1), a-b (2), b-d (3), d-v (4). 
    #          If b==c, then we have u-a-b-d-v. This is a C5 if d in N(u) or a in N(v)? 
    #          No, we are checking for C6. 
    #          If b==c, the path is u-a-b-d-v. This is a cycle of length 5 if (u,v) is added?
    #          Cycle: u-a-b-d-v-u. Length 5. 
    #          The task is C6-free. C5 is allowed. 
    #          So if b==c, we don't form a C6. We form a C5 (if u,v connected).
    #          So we only care about b != c.
    
    # Let's refine the check:
    # 1. Compute N_A_1: neighbors of all a in N(u).
    # 2. Compute N_B_1: neighbors of all d in N(v).
    # 3. Iterate through edges (b,c) in the graph.
    #    If b in N_A_1 and c in N_B_1:
    #       We have a path u-a-b-c-d-v.
    #       We must verify that the vertices are distinct:
    #       u, a, b, c, d, v.
    #       a is chosen from N(u) such that (a,b) in E.
    #       d is chosen from N(v) such that (d,c) in E.
    #       We need to ensure there exists such a and d such that:
    #       a != v, a != b, a != c, a != d, a != u (trivial)
    #       b != u, b != v, b != c, b != d, b != a
    #       c != u, c != v, c != b, c != d, c != a
    #       d != u, d != v, d != b, d != c, d != a
       
    # This is getting complex to verify distinctness on the fly.
    # Given n=40, we can just do a BFS of depth 4 from u in G \ {(u,v)} and see if we reach v?
    # Or simpler: 
    # For each a in N(u):
    #   For each b in N(a):
    #     if b == v: continue # length 2 path u-a-v? No, u-a-b-v is length 3.
    #     For each c in N(b):
    #       if c == v: continue # u-a-b-c-v is length 4. Adding (u,v) makes C5.
    #       if c in N(v):
    #          # We have u-a-b-c-d-v where d is a neighbor of c in N(v).
    #          # We need to check if we can pick d in N(v) such that d != u, d != a, d != b, d != c.
    #          # Also need a != b, b != c.
    #          # Since we iterate b in N(a), a != b is guaranteed (simple graph).
    #          # c in N(b), so b != c guaranteed.
    #          # So we just need to check if there exists d in N(v) such that:
    #          # (c,d) in E (implied by c in N(b) and d in N(c)? No, we check c in N(v) means d=c? No.)
    #          # Wait. The path is u-a-b-c-d-v.
    #          # c is connected to d. d is connected to v.
    #          # So d must be in N(c) AND d must be in N(v).
    #          # So we check if N(c) intersects N(v).
    #          # Let candidates_d = N(c) & N(v).
    #          # We need to find if there is a d in candidates_d such that:
    #          # d != u, d != a, d != b, d != c.
    
    if not nu or not nv:
        return False
        
    for a in nu:
        if a == v: continue
        for b in adj[a]:
            if b == u or b == a: continue
            if b == v: continue # Path u-a-b-v is length 3. C5 with (u,v).
            for c in adj[b]:
                if c == u or c == a or c == b: continue
                if c == v: continue # Path u-a-b-c-v is length 4. C5 with (u,v).
                # Check if c is connected to any d in N(v)
                # d must be in N(c) and N(v)
                # And d must not be u, a, b, c
                found_d = False
                for d in adj[c]:
                    if d == u or d == a or d == b or d == c: continue
                    if d in nv:
                        # Path u-a-b-c-d-v exists
                        # Check distinctness:
                        # u, a, b, c, d, v
                        # a != b (checked)
                        # b != c (checked)
                        # c != d (checked)
                        # a != c (checked)
                        # a != d? Not checked explicitly.
                        # d != a? 
                        # If d == a, path is u-a-b-c-a-v. Not a simple cycle.
                        # So we must check d != a.
                        if d == a: continue
                        # Also d != v is guaranteed since d in nv and nv are neighbors of v, but d could be v? No, simple graph no self loops.
                        # Also a != v checked.
                        # So if d is valid, we have a C6.
                        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_edges = []
    best_count = 0
    n = 40
    
    # Initial seed
    random.seed(42)
    
    # Run multiple randomized greedy constructions
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = [set() for _ in range(n)]
        edges = []
        
        # Create list of all possible edges
        all_edges = []
        for i in range(n):
            for j in range(i+1, n):
                all_edges.append((i, j))
        
        # Shuffle edges
        random.shuffle(all_edges)
        
        # Greedy addition
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if u in adj[v]:
                continue
            # Check if adding (u,v) creates C6
            if not has_c6_with_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    # If we have time, try local search on the best found
    if time.time() < deadline - 10 and best_edges:
        # Reconstruct adj from best_edges
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Simple local search: try to remove an edge and add two?
        # Or just try to add more edges if any were missed? 
        # The greedy should have added all possible.
        # So local search must involve swaps.
        # Try to find an edge e to remove, and two edges e1, e2 to add.
        # This is complex. 
        # Alternative: Randomly remove an edge, then run greedy again on remaining?
        # This is essentially a restart with a bias.
        
        # Let's just do more restarts if time permits.
        pass

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
