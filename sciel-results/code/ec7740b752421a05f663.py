import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    limit = 512
    n = limit
    
    # Build compatibility graph using bitmasks
    # adj[i] is a bitmask of vertices compatible with i
    # We use 0-indexed for bits, so vertex i corresponds to bit i-1? 
    # Or just use 1-indexed vertices and bit i? 
    # Python ints can handle 512 bits easily.
    # Let's use bit i for vertex i (1..512). So mask for v is (1 << v).
    
    adj = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            if i * j // g <= limit:
                # Compatible
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Tomita Maximum Clique Algorithm
    # Vertices are 1..n
    # P: set of candidates (bitmask)
    # X: set of excluded vertices (bitmask)
    # R: current clique (list)
    
    best_clique = []
    best_size = 0
    
    # Helper to get list of vertices from bitmask
    def bits_to_list(mask):
        res = []
        while mask:
            # Get least significant set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            res.append(idx)
            mask ^= lsb
        return res
    
    # MCS Coloring
    # Returns list of (color, vertex) sorted by color descending
    def mcs_coloring(P):
        # P is a bitmask
        if P == 0:
            return []
        
        # Get vertices in P
        # We need to order them by MCS
        # MCS: pick vertex with max degree in the induced subgraph of uncolored vertices?
        # Standard Tomita MCS:
        # 1. Order vertices in P by degree in P (descending) initially?
        #    Actually, MCS selects vertices one by one.
        #    Let's implement the standard "Coloring" step from Tomita et al.
        
        # We maintain a list of uncolored vertices.
        # At each step, pick the vertex with the highest degree among the uncolored ones?
        # Or just use a greedy approach on a static order?
        # Tomita's algorithm uses a specific ordering derived from MCS.
        
        # Let's use a simpler effective heuristic: 
        # 1. Get all vertices in P.
        # 2. Sort them by degree in P (descending).
        # 3. Greedily assign colors.
        # This is often sufficient for small graphs.
        
        verts = bits_to_list(P)
        # Calculate degrees within P
        degs = []
        for v in verts:
            # Count bits in adj[v] & P
            deg = (adj[v] & P).bit_count()
            degs.append(deg)
        
        # Sort by degree descending
        order = sorted(zip(verts, degs), key=lambda x: x[1], reverse=True)
        
        colors = {}
        max_color = 0
        # Greedy coloring
        # For each vertex in order, assign smallest available color not used by neighbors
        # Since we process in order, we only need to check neighbors that have already been colored
        # which are the ones earlier in the list.
        
        # To speed up, we can just iterate.
        # Note: This is O(V^2) in the worst case, but V <= 512.
        
        for v, _ in order:
            # Find used colors by neighbors in the already colored set
            used = set()
            # Neighbors in P
            neigh_mask = adj[v] & P
            # We need to check which neighbors are already colored
            # Since we don't have a "colored_mask" readily available in the loop without extra storage,
            # let's just check all neighbors in `order` that appear before v.
            # Actually, it's easier to just maintain a list of colors assigned so far.
            # But checking "is u colored?" is easy if we use a dict or set.
            
            # Let's do a slightly more efficient check:
            # We iterate through `order` again? No.
            # Let's just store assigned colors in a dict.
            
            # Optimization: Just check neighbors.
            # If we process in a fixed order, we can just look at the colors of neighbors that are in the prefix of the order.
            # Simpler: Just use a set of used colors for this vertex.
            
            # Get neighbors
            # We need to map neighbor vertex -> color
            # This is getting complex to optimize without arrays.
            # Given N=512, simple O(N^2) is fine.
            
            # Let's refine:
            # We have `order` list of (v, deg).
            # We will assign colors to `v` in this order.
            # For `v`, we look at all `u` in `order` that come before `v` and are neighbors of `v`.
            
            # To make this fast, we can precompute the index in `order` for each vertex?
            # Or just use a set of colored vertices.
            
            pass
        
        # Let's restart the coloring function with a simpler, robust implementation.
        # Since N is small, we can afford a bit of overhead.
        
        colored = {}
        # We need to assign colors such that no two adjacent vertices have the same color.
        # Greedy: for v in order:
        #   forbidden = { colored[u] for u in neighbors(v) if u in colored }
        #   color = min positive int not in forbidden
        #   colored[v] = color
        
        # To find neighbors in `colored`, we can iterate `colored` keys?
        # Or iterate `P` bits and check if in `colored`?
        
        # Let's do:
        for v, _ in order:
            forbidden = set()
            # Iterate through all neighbors of v in P
            neigh_mask = adj[v] & P
            # For each neighbor u, if u is in colored, add its color to forbidden
            # How to iterate neighbors?
            # We can iterate through `order` and check if neighbor is in colored?
            # That's O(N^2) per vertex -> O(N^3) total. 512^3 = 1.3e8. Might be too slow in Python.
            
            # Better: Iterate through `colored` items?
            # If `colored` is large, this is O(N^2) per vertex.
            
            # Let's use a different approach:
            # Maintain an array `color_of` of size N+1, initialized to 0.
            # And a list of colored vertices.
            
            pass
        
        # Let's implement the standard Tomita MCS properly.
        # It maintains a list `L` of vertices in P.
        # It picks a vertex, removes it, updates degrees.
        # This is complex to implement efficiently in Python with bitmasks.
        
        # Alternative: Use a simpler coloring that is still effective.
        # "Greedy coloring by degree order" is often good enough.
        
        # Let's try to optimize the "forbidden" calculation.
        # We can precompute the list of neighbors for each vertex as a list of ints.
        
        return [] # Placeholder

    # Let's rewrite the core solver more carefully.
    
    # Precompute neighbor lists for faster iteration if needed, 
    # but bitmasks are better for intersection.
    # For coloring, we need to check neighbors.
    
    # Let's stick to the bitmask intersection for the branching.
    # For coloring, we will use the "degree sorted" greedy method.
    
    def color_and_order(P):
        """
        Returns a list of (color, vertex) sorted by color descending.
        """
        if P == 0:
            return []
        
        verts = bits_to_list(P)
        
        # Calculate degrees in P
        # deg(v) = popcount(adj[v] & P)
        
        # Sort by degree descending
        # To avoid creating many small lists, we can just sort the verts list with a key
        # But we need degrees.
        
        # Let's compute degrees
        degs = {}
        for v in verts:
            degs[v] = (adj[v] & P).bit_count()
            
        # Sort vertices by degree descending
        order = sorted(verts, key=lambda x: degs[x], reverse=True)
        
        colors = {}
        max_color = 0
        
        # Greedy Coloring
        # We process vertices in `order`.
        # For each v, we determine the smallest color not used by its neighbors that are already colored.
        # "Already colored" means they appear earlier in `order`.
        
        # To check neighbors quickly:
        # We can iterate through `order` again? No.
        # We can iterate through the neighbors of v (using bitmask) and check if they are in `colors`.
        
        # Let's optimize the neighbor check.
        # Instead of checking all neighbors in P, we only care about those in `colors`.
        # So, we can intersect `adj[v]` with a bitmask of `colors` keys?
        # We don't have a bitmask for `colors` keys readily.
        
        # Let's just iterate through the items in `colors`?
        # If `colors` is large, this is slow.
        
        # Alternative:
        # For each v, get `neigh_mask = adj[v] & P`.
        # Then iterate through `order` (which is sorted by degree, but not necessarily by color assignment time... wait, it IS the assignment time).
        # So we can iterate through `order`, and for each `u`, if `u` is a neighbor of `v` and `u` is in `colors`, add `colors[u]` to forbidden.
        # This is O(N) per vertex. Total O(N^2).
        
        for v in order:
            forbidden = set()
            # Check all previously processed vertices (which are in colors)
            # We can just iterate through `colors` items?
            # Or iterate through `order` up to v?
            # Iterating `colors` is O(K) where K is number of colored vertices.
            # Total O(N^2).
            
            # Let's try iterating through `colors` items.
            for u, c in colors.items():
                if (adj[v] >> u) & 1: # Check if u is neighbor of v
                    forbidden.add(c)
            
            # Find smallest positive int not in forbidden
            c = 1
            while c in forbidden:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Create result list: (color, vertex)
        res = [(colors[v], v) for v in order]
        # Sort by color descending, then maybe by degree descending?
        res.sort(key=lambda x: x[0], reverse=True)
        
        return res

    def expand(P, X, R):
        nonlocal best_clique, best_size
        
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # Get colors and order
        colored_order = color_and_order(P)
        
        # colored_order is list of (color, vertex)
        # We process in this order.
        
        for color, v in colored_order:
            # Pruning
            if len(R) + color <= best_size:
                break
            
            # Branch on v
            # New P = P & adj[v]
            # New X = X & adj[v]
            # New R = R + [v]
            
            P_new = P & adj[v]
            X_new = X & adj[v]
            R.append(v)
            
            expand(P_new, X_new, R)
            
            R.pop()
            
            # Move v from P to X
            P = P & ~(1 << v)
            X = X | (1 << v)
            
            # Update best_size if R was a clique? 
            # No, we update at the leaf (P==0).
            # But we can also check if R is a maximal clique? 
            # The algorithm handles it.

    # Initial call
    # P is all vertices
    initial_P = 0
    for i in range(1, n + 1):
        initial_P |= (1 << i)
    
    expand(initial_P, 0, [])
    
    if not best_clique:
        # Fallback
        return [1]
        
    # The result is a list of vertices.
    # We need to return a sorted list.
    return sorted(best_clique)
