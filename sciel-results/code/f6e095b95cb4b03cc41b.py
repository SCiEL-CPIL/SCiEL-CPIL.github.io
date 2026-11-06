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
        
    # Initial best: see if we can find a large clique quickly
    # We will use a greedy heuristic to get a good initial lower bound
    best_clique = []
    
    # Greedy coloring and clique finding
    # We implement a standard Bron-Kerbosch with pivot and coloring
    
    # To make the coloring work effectively, we need to order vertices.
    # We can just use the natural order 0..n-1, or sort by degree.
    # Let's sort by degree descending to potentially find large cliques faster?
    # Actually, for Bron-Kerbosch, the order in P matters for the coloring heuristic.
    # We will maintain P as a list of vertices.
    
    # Let's use a simpler approach first: Bron-Kerbosch with pivot.
    # We represent P and X as sets or lists? 
    # Bitsets are faster for intersection, but iterating requires extracting bits.
    
    # Let's stick to lists for P and X for simplicity of the recursive structure, 
    # but use bitsets for adjacency checks.
    
    global_best = []
    
    def color_sort(vertices):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns a list of (vertex, color) pairs, sorted by color.
        This provides an upper bound on the clique size in any subset of these vertices.
        """
        colors = {}
        color_map = {}
        next_color = 1
        # Simple greedy: assign smallest possible color
        # We process vertices in the given order
        for v in vertices:
            # Find smallest color not used by neighbors of v in the already colored set
            used_colors = set()
            for u in vertices:
                if u == v:
                    continue
                if u in colors:
                    # Check if u is adjacent to v
                    if (adj[v] >> u) & 1:
                        used_colors.add(colors[u])
            
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > next_color:
                next_color = c
        
        # Sort by color
        sorted_vertices = sorted(vertices, key=lambda x: colors[x])
        return sorted_vertices, colors, next_color

    def expand(R, P, X):
        nonlocal global_best
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(global_best):
                global_best = R[:]
            return
        
        # If P is empty, we can't add anything
        if not P:
            return
            
        # Upper bound check using coloring
        # We color the vertices in P
        sorted_P, colors, max_color = color_sort(P)
        
        # If max_color + len(R) <= len(global_best), prune
        if max_color + len(R) <= len(global_best):
            return
            
        # Process vertices in reverse order of color (highest color first)
        # This is the standard Tomita approach
        for i in range(len(sorted_P) - 1, -1, -1):
            v = sorted_P[i]
            
            # If even with this vertex and all remaining higher colored vertices we can't beat best
            # Actually, the bound is: len(R) + (len(sorted_P) - i) <= len(global_best) is a loose bound.
            # The coloring bound is: len(R) + colors[v] <= len(global_best) ? No.
            # The bound is that the max clique in the subgraph induced by sorted_P[0..i] is at most colors[v] (if colors are assigned properly to be a valid coloring of that subgraph? No, the coloring is for the whole P).
            # Standard bound: if len(R) + i + 1 <= len(global_best), prune? No, that's just size.
            # The coloring bound is: The maximum clique size in P is at most max_color.
            # More specifically, the max clique in the subgraph induced by {v, ...} is bounded by the color assigned to v in the greedy coloring of the *remaining* candidates?
            # Let's use the property: In the sorted order, the max clique in P[0..i] is at most colors[sorted_P[i]].
            # So if len(R) + colors[sorted_P[i]] <= len(global_best), we can break the loop.
            
            if len(R) + colors[v] <= len(global_best):
                break
                
            # Add v to R
            R.append(v)
            
            # New P is P intersect neighbors of v
            new_P = [u for u in sorted_P[:i] if (adj[v] >> u) & 1]
            # New X is X intersect neighbors of v
            new_X = [u for u in X if (adj[v] >> u) & 1]
            
            expand(R, new_P, new_X)
            
            R.pop()
            
            # Move v to X
            X.append(v)
            
    # Initial call
    # Start with all vertices in P, X empty
    initial_P = list(range(n))
    initial_X = []
    initial_R = []
    
    # To improve initial bound, let's run a simple greedy clique finder first
    # Greedy: pick vertex with highest degree, add to clique, remove non-neighbors
    def greedy_clique():
        P = set(range(n))
        C = []
        while P:
            # Pick vertex with max degree in P
            v = max(P, key=lambda u: bin(adj[u] & ((1 << n) - 1)).count('1')) # This is slow
            # Better: just pick first one or one with high degree
            # Let's just pick the one with max degree in the current set P
            max_deg = -1
            best_v = None
            for u in P:
                # Count neighbors in P
                # adj[u] is a bitmask. We need to mask it with P's bitmask?
                # We don't have P as a bitmask here.
                # Let's convert P to a bitmask for speed? 
                # For n=512, iterating is okay.
                deg = 0
                for w in P:
                    if u != w and ((adj[u] >> w) & 1):
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_v = u
            C.append(best_v)
            # Remove non-neighbors of best_v from P
            P = {u for u in P if u != best_v and ((adj[best_v] >> u) & 1)}
        return C
    
    g_c = greedy_clique()
    if len(g_c) > len(global_best):
        global_best = g_c[:]
        
    expand(initial_R, initial_P, initial_X)
    
    # If global_best is still empty (shouldn't happen, at least {1} is a clique), fallback
    if not global_best:
        return [1]
        
    return global_best

