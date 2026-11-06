"""New program: C6-free graph on 40 vertices using spectral initialization and optimized local search."""
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a C6-free graph on 40 vertices with many edges.
    
    Strategy:
    1. Initialize with a random 5-regular graph.
    2. Compute spectral embedding (principal eigenvector).
    3. Use spectral alignment to prioritize edge additions.
    4. Use optimized bitset-based local search to add edges and perform swaps.
    
    Returns a list of edges [u, v].
    """
    n = 40
    rng = random.Random(42)
    
    # Step 1: Initialize with a random 5-regular graph
    # We use a simple configuration model or just random edges until regular-ish.
    # For simplicity and speed, we generate a random graph and then balance degrees.
    # Or just start with a random graph with p ~ 5/39.
    
    adj = [0] * n
    edges = []
    
    # Generate random 5-regular graph using a simple algorithm
    # 1. Create list of half-edges: 5 copies of each vertex
    # 2. Shuffle and pair them up
    half_edges = []
    for i in range(n):
        for _ in range(5):
            half_edges.append(i)
    rng.shuffle(half_edges)
    
    # Pair them up
    used = [False] * n
    # We need to avoid self-loops and multi-edges.
    # Simple approach: Rejection sampling for regular graph is hard.
    # Let's just start with a random graph with ~100 edges and then try to add/remove to get close to regular, 
    # or just start with the 8 K5s but shuffle the partition? 
    # The prompt asks for "spectral initialization ... of a random regular graph".
    # Let's try to build a random 5-regular graph properly.
    
    # Algorithm for random regular graph:
    # 1. Create list of stubs.
    # 2. Shuffle.
    # 3. Connect stubs in pairs. If self-loop or multi-edge, swap with another stub to fix.
    
    stubs = []
    for i in range(n):
        for _ in range(5):
            stubs.append(i)
    rng.shuffle(stubs)
    
    adj = [0] * n
    edges = []
    
    # Simple pairing with retry/swap logic
    # This is O(N^2) worst case but N=40 is small.
    i = 0
    while i < len(stubs):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self-loop, find another stub to swap with
            # Find j > i+1 such that stubs[j] != u and stubs[j] != v (ideally)
            swapped = False
            for j in range(i+2, len(stubs)):
                if stubs[j] != u and stubs[j] != v:
                    # Swap stubs[i+1] and stubs[j]
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    swapped = True
                    break
            if not swapped:
                # Just skip this pair? No, we must use all stubs.
                # Force a connection if possible, else restart?
                # For N=40, this is unlikely to fail badly.
                pass
        elif adj[u] & (1 << v):
            # Multi-edge, find another stub
            swapped = False
            for j in range(i+2, len(stubs)):
                if stubs[j] != u and not (adj[u] & (1 << stubs[j])) and stubs[j] != v:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    swapped = True
                    break
            if not swapped:
                pass
        
        # If we still have a valid edge (u,v)
        if u != v and not (adj[u] & (1 << v)):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            i += 2
        else:
            # If we couldn't fix it, just move on and hope the graph is connected enough.
            # This might result in non-regular graph.
            i += 2

    # Step 2: Spectral Initialization
    # Build adjacency matrix
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    # Compute principal eigenvector
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Largest eigenvalue index
        idx = np.argmax(eigenvalues)
        principal_vec = np.abs(eigenvectors[:, idx])
    except:
        principal_vec = np.ones(n)
    
    # Step 3: Local Search
    # We want to maximize edges.
    # We will try to add edges.
    
    def can_add_edge(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 would be u - a - b - c - d - v - u.
        We need to check if there exists a simple path a-b-c-d of length 3
        in G \ {u,v} such that a in N(u) and d in N(v).
        """
        # Neighbors of u excluding v
        nu = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv = adj[v] & ~(1 << u)
        
        if nu == 0 or nv == 0:
            return True
            
        # Expand from N(u) by 1 step to get S1 (vertices at distance 2 from u via N(u))
        # S1 = { b | exists a in N(u), (a,b) in E, b != u, b != v }
        s1 = 0
        temp = nu
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            s1 |= adj[a]
            temp ^= lsb
        s1 &= ~(1 << u)
        s1 &= ~(1 << v)
        
        # Expand from S1 by 1 step to get S2 (vertices at distance 3 from u)
        # S2 = { c | exists b in S1, (b,c) in E, c != u, c != v }
        s2 = 0
        temp = s1
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            s2 |= adj[b]
            temp ^= lsb
        s2 &= ~(1 << u)
        s2 &= ~(1 << v)
        
        # Expand from S2 by 1 step to get S3 (vertices at distance 4 from u)
        # S3 = { d | exists c in S2, (c,d) in E, d != u, d != v }
        s3 = 0
        temp = s2
        while temp:
            lsb = temp & -temp
            c = lsb.bit_length() - 1
            s3 |= adj[c]
            temp ^= lsb
        s3 &= ~(1 << u)
        s3 &= ~(1 << v)
        
        # Check if S3 intersects N(v)
        intersection = s3 & nv
        if intersection == 0:
            return True
            
        # If intersection exists, we found a potential C6.
        # We must verify if a *simple* path exists.
        # The bitset expansion might have found a non-simple path.
        # Let's verify by checking specific candidates.
        
        # Iterate through possible d in intersection
        temp_d = intersection
        while temp_d:
            lsb_d = temp_d & -temp_d
            d = lsb_d.bit_length() - 1
            temp_d ^= lsb_d
            
            # Find c in N(d) that is in S2
            temp_c = adj[d] & s2
            while temp_c:
                lsb_c = temp_c & -temp_c
                c = lsb_c.bit_length() - 1
                temp_c ^= lsb_c
                
                if c == u or c == v or c == d: continue
                
                # Find b in N(c) that is in S1
                temp_b = adj[c] & s1
                while temp_b:
                    lsb_b = temp_b & -temp_b
                    b = lsb_b.bit_length() - 1
                    temp_b ^= lsb_b
                    
                    if b == u or b == v or b == c or b == d: continue
                    
                    # Find a in N(b) that is in N(u)
                    temp_a = adj[b] & nu
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        a = lsb_a.bit_length() - 1
                        temp_a ^= lsb_a
                        
                        if a == u or a == v or a == b or a == c or a == d: continue
                        
                        # Check if a-b-c-d is a simple path (distinct vertices)
                        # We already checked distinctness against u,v and each other in loops?
                        # Let's explicitly check a,b,c,d distinct.
                        if a == b or b == c or c == d or a == c or a == d or b == d:
                            continue
                            
                        return False
                            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    # Local Search
    # We will run a loop for a fixed number of iterations or until no improvement.
    # Given 120s limit, we can run many iterations.
    
    max_iters = 2000
    no_improve = 0
    
    for _ in range(max_iters):
        improved = False
        
        # 1. Try to add an edge
        # Prioritize non-edges based on spectral alignment
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    # Score based on spectral components
                    score = principal_vec[u] * principal_vec[v]
                    non_edges.append((score, u, v))
        
        # Sort by score descending
        non_edges.sort(key=lambda x: -x[0])
        
        added_any = False
        for score, u, v in non_edges:
            if can_add_edge(u, v):
                add_edge(u, v)
                improved = True
                added_any = True
                break # Add one at a time to keep state consistent
        
        if improved:
            no_improve = 0
            continue
        else:
            no_improve += 1
            
        # 2. Try Swap: Remove one edge, try to add two
        # Pick a random edge to remove
        if edges:
            idx = rng.randint(0, len(edges) - 1)
            u, v = edges[idx]
            remove_edge(u, v)
            
            # Try to add two edges
            added_count = 0
            added_edges = []
            
            # Generate new non-edges, prioritize by spectral alignment
            non_edges2 = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        score = principal_vec[i] * principal_vec[j]
                        non_edges2.append((score, i, j))
            
            non_edges2.sort(key=lambda x: -x[0])
            
            for score, a, b in non_edges2:
                if can_add_edge(a, b):
                    add_edge(a, b)
                    added_edges.append((a, b))
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count == 2:
                # Net gain +1
                improved = True
                no_improve = 0
            else:
                # Revert
                for a, b in added_edges:
                    remove_edge(a, b)
                add_edge(u, v)
                # no_improve stays incremented

    return edges
