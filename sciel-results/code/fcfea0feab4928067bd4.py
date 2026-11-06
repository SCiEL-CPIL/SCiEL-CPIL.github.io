import time
import random
import numpy as np

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def build_bitsets(adj):
        return [sum(1 << j for j in range(n) if adj[i][j]) for i in range(n)]

    def has_c6(adj, bs):
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                # Look for path i-a-b-c-j of length 4
                # a in N(i), c in N(j), b in N(a) ∩ N(c)
                # b != i, j, a, c
                # a != c, a != j, c != i
                common = bs[i] & bs[j]
                # We need a in N(i), c in N(j) such that N(a) & N(c) has a bit b not in {i,j,a,c}
                # Iterate a in N(i)
                ai = bs[i] & ~(1 << i)
                aj = bs[j] & ~(1 << j)
                # For efficiency, we can just check if there exists a, c
                # This is O(d^2 * n) per edge, which is slow.
                # Better: For each edge (i,j), check if there is a path of length 4.
                # Actually, let's use the standard C6 check:
                # For each pair (i,j) not adjacent, check if dist(i,j) == 4? No, C6 is a cycle.
                # A C6 is a set of 6 vertices v0..v5 with edges (v0,v1)..(v5,v0).
                # To check if graph has C6, we can iterate over pairs of non-adjacent vertices (i,j)
                # and check if there is a path of length 4 between them.
                # Path i-a-b-c-j.
                # a in N(i), c in N(j), b in N(a) & N(c).
                # b != i, j, a, c.
                # a != c.
                # Let's optimize this.
                pass
        return False

    # A faster C6 check using bitsets
    def check_c6_fast(adj, bs):
        # For each edge (i,j), we want to know if there is a path of length 4 between i and j
        # that does not use the edge (i,j).
        # Path i-a-b-c-j.
        # a in N(i), c in N(j).
        # b in N(a) & N(c).
        # b != i, j, a, c.
        # a != c.
        # Note: a and c are neighbors of i and j respectively.
        # If a == c, then we have a triangle i-a-j, which is C3, not C6.
        # So we require a != c.
        # Also b != i, j, a, c.
        
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                # Remove i and j from neighbor sets to avoid using the edge directly?
                # No, the path is i-a-b-c-j. The edge (i,j) is the closing edge.
                # The path is of length 4.
                # a in N(i), c in N(j).
                # We need to check if there exists a, c such that a != c and (N(a) & N(c)) has a bit b
                # such that b is not in {i, j, a, c}.
                
                # Let's iterate a in N(i)
                # To speed up, we can precompute for each pair (a,c) if they have a common neighbor
                # that is not in {i,j,a,c}.
                
                # Simple approach:
                found = False
                ai = bs[i]
                aj = bs[j]
                # a in N(i), c in N(j)
                # We can iterate a in N(i)
                temp_ai = ai
                while temp_ai:
                    lsb = temp_ai & -temp_ai
                    a = lsb.bit_length() - 1
                    temp_ai ^= lsb
                    if a == j: continue # a cannot be j because i-j is edge, but a is neighbor of i.
                    # c in N(j), c != a
                    temp_aj = aj & ~(1 << a)
                    while temp_aj:
                        lsb_c = temp_aj & -temp_aj
                        c = lsb_c.bit_length() - 1
                        temp_aj ^= lsb_c
                        if c == i: continue
                        # Check common neighbors of a and c
                        common = bs[a] & bs[c]
                        # Remove i, j, a, c from common
                        common &= ~(1 << i)
                        common &= ~(1 << j)
                        common &= ~(1 << a)
                        common &= ~(1 << c)
                        if common:
                            return True
        return False

    def can_add_edge(u, v, adj, bs):
        # Adding (u,v) creates C6 if there is a path u-x-y-z-v of length 4.
        # x in N(u), z in N(v), y in N(x) & N(z).
        # y != u, v, x, z.
        # x != z.
        
        # Iterate x in N(u)
        temp_x = bs[u]
        while temp_x:
            lsb = temp_x & -temp_x
            x = lsb.bit_length() - 1
            temp_x ^= lsb
            if x == v: continue
            # Iterate z in N(v)
            temp_z = bs[v] & ~(1 << x)
            while temp_z:
                lsb_z = temp_z & -temp_z
                z = lsb_z.bit_length() - 1
                temp_z ^= lsb_z
                if z == u: continue
                # Check common neighbors of x and z
                common = bs[x] & bs[z]
                common &= ~(1 << u)
                common &= ~(1 << v)
                common &= ~(1 << x)
                common &= ~(1 << z)
                if common:
                    return False
        return True

    # Spectral Initialization
    # Generate a random regular graph (e.g., 5-regular)
    # Compute eigenvector corresponding to eigenvalue closest to sqrt(5)
    # Use this to guide edge addition.
    
    # For simplicity and speed, we will use a few random restarts with a different initial graph
    # generated by spectral method.
    
    best_edges = []
    best_count = 0
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # 1. Generate random 5-regular graph on 40 vertices
        # Using configuration model or simple random matching
        # Since n=40 is small, we can use numpy random permutation
        # 5-regular means each vertex has degree 5. Total edges = 100.
        
        # We'll just use a random graph with p=0.25 (expected degree 9.75) and then prune?
        # Or stick to the K5s base? The prompt says "spectral initialization".
        
        # Let's create a random graph G0.
        # To be spectral, we need eigenvalues.
        # Let's generate a random 40x40 symmetric matrix with 0/1 entries, no loops.
        # Then compute eigenvectors.
        
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    A[i,j] = 1
                    A[j,i] = 1
        
        # Compute eigenvalues and eigenvectors
        # This might be slow? 40x40 is fine.
        try:
            evals, evecs = np.linalg.eigh(A)
        except:
            continue
            
        # Find eigenvector with eigenvalue closest to sqrt(5) ~ 2.236
        idx = np.argmin(np.abs(evals - 2.236))
        v = np.abs(evecs[:, idx])
        
        # Use v to guide edge addition.
        # Probability of edge (i,j) is proportional to v[i] * v[j]
        # We want to add edges to maximize count while staying C6-free.
        
        adj = [[0]*n for _ in range(n)]
        edges = []
        
        # Initial step: Add edges with high probability based on spectral components
        # But we must ensure no C6.
        # We will greedily add edges in order of decreasing v[i]*v[j]
        
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                prob = v[i] * v[j]
                candidates.append((prob, i, j))
                
        candidates.sort(reverse=True)
        
        bs = build_bitsets(adj)
        
        for prob, i, j in candidates:
            if time.time() > deadline - 15:
                break
            if adj[i][j]:
                continue
            if can_add_edge(i, j, adj, bs):
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
                bs[i] |= (1 << j)
                bs[j] |= (1 << i)
                
        # Local Search: Try to add more edges or swap
        # Since we added greedily, we might have missed some.
        # Let's do a pass to add any remaining valid edges.
        added = True
        while added and time.time() < deadline - 15:
            added = False
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        continue
                    if can_add_edge(i, j, adj, bs):
                        adj[i][j] = 1
                        adj[j][i] = 1
                        edges.append([i, j])
                        bs[i] |= (1 << j)
                        bs[j] |= (1 << i)
                        added = True
                        if time.time() > deadline - 15:
                            break
                if time.time() > deadline - 15:
                    break
                    
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            if best_count >= 105:
                break

    return best_edges
