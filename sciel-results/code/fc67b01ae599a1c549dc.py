import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    # Vertices 0..511 correspond to numbers 1..512
    
    # Precompute LCMs and compatibility
    # adj[i] is a bitset of vertices compatible with i
    adj = [0] * n
    
    # Optimization: Compute compatibility
    for i in range(n):
        num_i = i + 1
        bits = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            # lcm(i, j) = i * j // gcd(i, j)
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= n:
                bits |= (1 << j)
        adj[i] = bits
        
    # Initial best: seed size 22
    best_size = 22
    best_set = list(range(1, 23))
    
    # Tomita's Algorithm for Maximum Clique
    # We represent the set of candidates P as a bitset (integer)
    
    # Order vertices by degree (number of compatible neighbors)
    # Degree of i is popcount(adj[i])
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    # Sort indices by descending degree
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We will map the sorted order to 0..n-1 for easier bit manipulation?
    # Or just use the original indices and iterate in 'order'.
    # Using original indices is easier for adj lookup.
    
    # Recursive search
    # P: bitset of candidates
    # C: current clique size
    # 
    # To implement the coloring bound efficiently in Python recursion:
    # We can compute the coloring on the fly for the current P.
    
    def popcount(x):
        return bin(x).count('1')
    
    def greedy_color_bound(P):
        # Returns (max_color, colors_list) where colors_list[i] is color of vertex i in P?
        # Actually, we just need the max color for the whole set, 
        # but for pruning in the loop, we need the color of each vertex.
        # Let's return a list of (vertex, color) sorted by color descending.
        
        U = P
        colors = {}
        color = 0
        
        # We need to pick an independent set from U repeatedly.
        # A simple way:
        # Iterate through vertices in U.
        # This is O(n^2) per node in worst case, which might be slow.
        
        # Alternative:
        # Just assign colors sequentially.
        # But we need a tight bound.
        
        # Let's try a simpler heuristic for the bound:
        # Max degree in P + 1? No, too loose.
        
        # Let's implement the standard greedy coloring:
        # 1. Get list of vertices in P.
        # 2. Sort them by degree in P (descending).
        # 3. Assign colors greedily.
        
        verts = []
        tmp = P
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            verts.append(v)
            tmp ^= lsb
            
        # Calculate degrees in subgraph P
        degs = {}
        for v in verts:
            # Count neighbors in P
            degs[v] = popcount(adj[v] & P)
            
        # Sort by degree descending
        verts.sort(key=lambda x: degs[x], reverse=True)
        
        # Greedy coloring
        col = {}
        next_color = 1
        for v in verts:
            # Find smallest color not used by neighbors
            used = set()
            for u in verts:
                if u == v: break # Optimization: only check previous? No, neighbors can be any.
                # Actually, in sequential coloring, we only need to check already colored neighbors.
                # But here we are assigning in order.
                # So we check neighbors that have been assigned a color already.
                if u in col and (adj[v] >> u) & 1:
                    used.add(col[u])
            # Find smallest available
            c = 1
            while c in used:
                c += 1
            col[v] = c
            if c > next_color:
                next_color = c
                
        return next_color, col, verts

    # This per-node coloring is too slow for deep recursion in Python if the graph is large.
    # However, n=512 is small. The branching factor might be high.
    
    # Let's try a simpler approach: 
    # Use the initial coloring of the whole graph? No, P changes.
    
    # Given the time limit, a simpler greedy might be safer if exact is too slow.
    # But the task asks for maximum.
    # Let's try to optimize the exact solver.
    
    # Alternative: Use a library? No, standard library only.
    
    # Let's try a different strategy:
    # The graph is dense?
    # If $a|b$, they are connected.
    # Most numbers are connected to many others.
    # The complement (incompatibility) graph is sparse?
    # lcm(a,b) > 512.
    # If a and b are large, lcm is likely large.
    # So the incompatibility graph is dense among large numbers.
    # The compatibility graph is dense among small numbers.
    # We want a clique in compatibility graph.
    # This is likely to be found among small numbers.
    
    # Let's run a greedy heuristic first to get a good lower bound.
    # Then use branch and bound.
    
    # Greedy:
    # Start with all vertices.
    # Repeatedly pick the vertex with highest degree, add to clique, remove it and its non-neighbors?
    # No, for Max Clique:
    # 1. Start with P = all vertices.
    # 2. Pick v in P with max degree.
    # 3. Add v to Clique.
    # 4. P = P intersect neighbors(v).
    # 5. Repeat until P empty.
    
    def greedy_clique():
        P = (1 << n) - 1
        clique = []
        while P:
            # Find max degree in P
            best_v = -1
            best_deg = -1
            tmp = P
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & P)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                tmp ^= lsb
            clique.append(best_v)
            P = P & adj[best_v]
        return clique
    
    g_clique = greedy_clique()
    if len(g_clique) > best_size:
        best_size = len(g_clique)
        best_set = [v+1 for v in g_clique]
        
    # Now Branch and Bound
    # We need a faster coloring bound.
    # Maybe we can just use the size of P as a bound initially? No, too weak.
    
    # Let's try a limited depth search or a different heuristic.
    # Given 512 nodes, and the structure, the optimal solution is likely close to the greedy one.
    # If greedy gives 25, maybe optimal is 25 or 26.
    
    # Let's implement a simple recursive search with a time check.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(P, C, clique):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if C > best_size:
                best_size = C
                best_set = clique[:]
            return
            
        # Bound: C + popcount(P) <= best_size
        if C + popcount(P) <= best_size:
            return
            
        # Simple coloring bound is expensive.
        # Let's try to order vertices by degree in P and iterate.
        verts = []
        tmp = P
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            verts.append(v)
            tmp ^= lsb
            
        # Sort by degree in P
        verts.sort(key=lambda x: popcount(adj[x] & P), reverse=True)
        
        for v in verts:
            if time.time() > deadline:
                return
            # Pruning: if C + 1 + popcount(P & adj[v]) <= best_size, skip?
            # This is a weak bound.
            
            new_P = P & adj[v]
            search(new_P, C + 1, clique + [v])
            
            # Remove v from P for subsequent iterations
            P = P & ~(1 << v)
            
            if C + popcount(P) <= best_size:
                break

    # Start search with all vertices
    initial_P = (1 << n) - 1
    search(initial_P, 0, [])
    
    return best_set
