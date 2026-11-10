import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # 1. Precompute LCMs and find bad triples
    # lcm_table[i][j] for 1 <= i <= j <= n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Find all bad triples (a, b, c) with a < b < c
    # such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We need to build the conflict graph.
    # Two vertices u, v are connected if they appear together in any bad triple.
    
    # To efficiently find bad triples:
    # For each pair (a, b), let L = lcm(a, b).
    # We need to find c > b such that lcm(a, c) == L and lcm(b, c) == L.
    # Note: lcm(a, c) == L implies c | L and gcd(c, a) = c * gcd(a, L) / L ? No.
    # lcm(a, c) = L <=> L % a == 0 and L % c == 0 and lcm(a, c) = L.
    # Actually, lcm(a, c) = L implies c is a divisor of L, and L / gcd(a, c) == a * (L / a) / (L / gcd(a,c))?
    # Simpler: lcm(a, c) = L iff c | L and a | L and gcd(a, c) == a * c / L.
    # Or simply check lcm(a, c) == L for c | L.
    
    # Let's iterate a < b < c directly but optimized.
    # Number of triples is O(n^3) = 128^3 ~ 2M. This is fast enough in Python if optimized?
    # 2M iterations might take ~0.2s - 1s. Acceptable.
    
    # However, we can be smarter.
    # For each L, let D_L be the set of divisors of L in [1, 128].
    # Any bad triple must have lcm = L.
    # So for each L, we look at triples (a,b,c) from D_L such that lcm(a,b)=L, lcm(a,c)=L, lcm(b,c)=L.
    # This is still complex. Let's stick to the direct O(n^3) check but with early breaks or vectorization if possible?
    # Python loops are slow. 2M iterations is okay.
    
    # Let's use a boolean matrix for the graph.
    # graph[i][j] = True if i, j are in a bad triple.
    # We can use bitmasks for the graph to speed up B&B.
    
    # Initialize graph as list of integers (bitmasks)
    # graph_mask[i] is a bitmask of vertices connected to i.
    graph_mask = [0] * (n + 1)
    
    # We need to identify all bad triples.
    # To avoid O(n^3) in pure Python which might be slow (2M * constant), 
    # let's try to optimize.
    # Actually, 128^3 is 2,097,152. In Python, a simple loop might take 0.5-1.0s.
    # This is within the 85s budget.
    
    # We will store bad pairs in a set to build the graph.
    # Using a set of tuples (min, max) might be slower than a 2D bool array or bitmask updates.
    # Let's update graph_mask directly.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            # c must be > b
            # And lcm(a, c) == lab and lcm(b, c) == lab
            # c must be a divisor of lab.
            # Also c > b.
            # Iterate c from b+1 to n?
            # Or iterate divisors of lab?
            # Number of divisors of lab is small (max ~100 for numbers < 10000).
            # lab can be up to 128*127 ~ 16k. Divisors are few.
            
            # Find divisors of lab that are > b and <= n
            # And check lcm conditions.
            
            # To get divisors of lab efficiently, we can precompute divisors for all possible LCMs?
            # Max LCM is 128*127 = 16256.
            # Let's precompute divisors for all numbers up to 16256? No, just for lab values encountered.
            # Or just iterate c from b+1 to n?
            # 128^3 / 6 ~ 350k iterations. This is very fast.
            
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    # Found bad triple (a, b, c)
                    # Add edges (a,b), (a,c), (b,c)
                    graph_mask[a] |= (1 << b)
                    graph_mask[b] |= (1 << a)
                    
                    graph_mask[a] |= (1 << c)
                    graph_mask[c] |= (1 << a)
                    
                    graph_mask[b] |= (1 << c)
                    graph_mask[c] |= (1 << b)
                    
                    # Early exit? No, we need to mark all edges.
                    # But we can break if we just want to know existence? No, we need the full graph.
                    # However, once we find one c, we know (a,b) is an edge.
                    # We still need to find other c's to mark (a,c) and (b,c).
                    # So we cannot break on first c.
                    
    # 2. Branch and Bound for Maximum Independent Set
    # Vertices 1..128. Index 0 is unused.
    # We use bitmasks.
    # P: candidates
    # X: current solution (not needed for mask, just count)
    
    best_size = 0
    best_set_mask = 0
    
    # Initial P is all vertices
    initial_P = 0
    for i in range(1, n + 1):
        initial_P |= (1 << i)
        
    # Tomita's algorithm
    # We need a greedy coloring of the subgraph induced by P.
    # Returns list of colors, and the max color.
    
    def greedy_color(P_mask):
        # Returns (colors_dict, max_color)
        # colors_dict maps vertex to color
        # We process vertices in decreasing order of degree in P?
        # Or just fixed order? Tomita uses a specific ordering.
        # Let's use a simple heuristic: order by degree in P.
        
        # Get list of vertices in P
        verts = []
        temp = P_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            verts.append(v)
            temp &= temp - 1
            
        # Sort by degree in P (descending)
        # Degree of v in P is popcount(graph_mask[v] & P_mask)
        def deg(v):
            return bin(graph_mask[v] & P_mask).count('1')
            
        verts.sort(key=deg, reverse=True)
        
        colors = {}
        max_color = 0
        # Available colors for each vertex?
        # We assign colors greedily.
        # color[v] = smallest positive integer not used by neighbors of v already colored.
        
        for v in verts:
            # Find neighbors of v that are already colored
            # Neighbors in P: graph_mask[v] & P_mask
            # Colored neighbors: we need to know which neighbors have been assigned colors.
            # Since we iterate in a specific order, we can maintain a set of used colors by neighbors.
            
            used_colors = set()
            neigh_mask = graph_mask[v] & P_mask
            # Iterate through colored neighbors?
            # It's faster to iterate through the vertices we have already processed.
            # But we don't have a direct "colored" mask separate from P?
            # Actually, all vertices in verts are in P.
            # We process them one by one.
            # So "colored neighbors" are those in `verts` that appear before `v` in the sorted list?
            # No, the sorted list is the processing order.
            
            # Let's just iterate through all previously processed vertices?
            # That's O(V^2). V=128. 128^2 = 16k. Very fast.
            
            # Better: maintain a list of processed vertices.
            # For current v, check all processed vertices u.
            # If (u, v) is an edge, add color[u] to used_colors.
            
            # To make this fast, we can just loop through the `verts` list up to current index.
            # But `verts` is sorted.
            
            # Let's re-implement:
            # We have the list `verts` sorted by degree.
            # We iterate `for i, v in enumerate(verts):`
            # `used_colors` is empty initially for each v.
            # We need to check neighbors `u` of `v` that are in `verts[0:i]`.
            
            # This is O(V^2) per coloring step.
            # Number of coloring steps is number of nodes in B&B tree.
            # Could be large.
            
            # Optimization:
            # Instead of checking all previous vertices, we can maintain an array `vertex_color`.
            # And for each v, we look at its neighbors in P.
            # `neigh_mask = graph_mask[v] & P_mask`.
            # We need the colors of neighbors that have already been colored.
            # This is tricky because the coloring is greedy and sequential.
            
            # Standard Tomita implementation:
            # The vertices are ordered.
            # Color of v is the smallest positive integer not in { color[u] | u is neighbor of v and u is colored }.
            
            # Let's use the simple O(V^2) check for now, as V=128 is small.
            
            pass # Placeholder, will implement in helper
            
        return colors, max_color

    # Let's write the B&B function directly.
    
    def popcount(x):
        return bin(x).count('1')
        
    def expand(P_mask, X_count):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if X_count > best_size:
                best_size = X_count
                # We need to reconstruct the set.
                # Storing the set in the recursion is expensive.
                # We can store the path or just return the mask.
                # For now, let's just track size. If we need the set, we can do a second pass or store it.
                # The task asks to return the subset.
                # So we need to track the actual set.
                # Storing a bitmask for X is better.
                pass
            return
            
        # Upper bound: X_count + max possible size of independent set in P
        # Use greedy coloring bound.
        
        # Get vertices in P
        # Order them by degree in P (descending)
        verts = []
        temp = P_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            verts.append(v)
            temp &= temp - 1
            
        if not verts:
            return
            
        # Sort by degree in P
        def deg_in_P(v):
            return popcount(graph_mask[v] & P_mask)
            
        verts.sort(key=deg_in_P, reverse=True)
        
        # Greedy Coloring
        # We assign colors to vertices in the order of `verts`.
        # color[v] is the smallest positive integer not used by neighbors of v that are already colored.
        
        colors = {}
        max_color = 0
        # To speed up finding "used colors by neighbors", we can maintain a list of colored vertices.
        # But checking all previous vertices is O(V^2).
        
        for v in verts:
            used = set()
            # Check neighbors of v that are in P and already colored
            # Neighbors in P: graph_mask[v] & P_mask
            # We need to intersect this with the set of vertices processed so far.
            # Since we are iterating `verts` in order, "processed so far" is `verts[:i]`.
            # It is easier to just iterate through all `u` in `verts` that are neighbors of `v`.
            # But we only care about those already colored.
            
            # Alternative:
            # For each v, look at its neighbors in P.
            # For each neighbor u, if u is in `colors`, add colors[u] to `used`.
            
            neigh_mask = graph_mask[v] & P_mask
            # Iterate neighbors
            temp_neigh = neigh_mask
            while temp_neigh:
                u = (temp_neigh & -temp_neigh).bit_length() - 1
                if u in colors:
                    used.add(colors[u])
                temp_neigh &= temp_neigh - 1
                
            # Find smallest positive integer not in used
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        # The bound is X_count + max_color
        if X_count + max_color <= best_size:
            return
            
        # Branch on vertices
        # We process vertices in order of decreasing color (or just the order `verts`?)
        # Tomita suggests branching on vertices in the order they are colored?
        # Usually, we branch on the first vertex in the sorted order (highest degree).
        # But the bound relies on the coloring.
        # Standard implementation:
        # For each vertex v in `verts` (sorted by degree):
        #   if X_count + colors[v] <= best_size: break (since subsequent vertices have lower or equal color? No, colors are not monotonic with degree necessarily, but the bound for the rest is bounded by the max color of the remaining? No.)
        
        # Actually, the bound `max_color` is for the whole set `P`.
        # When we branch on `v`, we split `P` into `P \ {v}` and `P \ N[v]`.
        
        # Let's iterate through `verts` in reverse order of color?
        # Or just iterate through `verts`.
        
        # A common variant:
        # Order vertices by color (descending).
        # For i from len(verts)-1 down to 0:
        #   v = verts[i]
        #   if X_count + colors[v] <= best_size: break
        #   # Branch 1: Include v
        #   # Branch 2: Exclude v
        
        # Let's sort `verts` by color descending for branching.
        # But the coloring was done on the original `verts` order.
        # The colors are assigned based on the degree order.
        # We can create a list `branch_order` sorted by `colors[v]` descending.
        
        branch_order = sorted(verts, key=lambda x: colors[x], reverse=True)
        
        for v in branch_order:
            if X_count + colors[v] <= best_size:
                break
                
            # Branch 1: Include v
            # New P = P \ (N[v])
            # N[v] is v and its neighbors in P.
            new_P1 = P_mask & ~(graph_mask[v] | (1 << v))
            # New X_count = X_count + 1
            
            # We need to pass the current set X to reconstruct the solution.
            # To save memory/time, we can just pass the mask of X.
            # But X is growing.
            # Let's pass `current_X_mask`.
            
            # Recurse
            expand(new_P1, X_count + 1)
            
            if time.time() > deadline:
                return
            
            # Branch 2: Exclude v
            # New P = P \ {v}
            # We update P_mask for subsequent iterations in this loop?
            # Yes, we remove v from P for the next iterations.
            P_mask &= ~(1 << v)
            
            # The bound for the remaining vertices is still valid?
            # The coloring was done for the original P.
            # When we remove v, the bound `colors[v]` for the remaining set is not directly `max_color`.
            # However, the standard Tomita algorithm updates the bound.
            # Actually, the condition `X_count + colors[v] <= best_size` uses the color of v in the *current* coloring.
            # But we are removing v. The remaining vertices have colors assigned in the original coloring.
            # Is it safe to break if `X_count + colors[v] <= best_size`?
            # Yes, because `colors[v]` is an upper bound on the size of an independent set in the subgraph induced by `verts[i:]`?
            # No, `colors[v]` is just the color of v.
            # The bound for the remaining set `P'` is `max(color[u] for u in P')`.
            # Since we process in decreasing color, `colors[v]` is the max color of the remaining vertices.
            # So `X_count + colors[v]` is a valid upper bound for the branch where v is excluded (and subsequent ones).
            
            # So the loop structure is correct.
            
            # Wait, I need to handle the "Include v" branch correctly.
            # When I include v, I recurse on `new_P1`.
            # When I exclude v, I continue the loop with `P_mask` updated.
            
            # One issue: `colors` was computed for the original `P`.
            # When I remove v, the colors of other vertices don't change.
            # But the bound `colors[v]` for the remaining set is correct because the remaining set is a subset of the original, and we are processing in decreasing color order.
            
            # Let's refine the "Include v" recursion.
            # We need to know the current set X to return it.
            # I will modify `expand` to accept `current_X_mask`.
            
    # Revised Expand
    def expand(P_mask, X_count, X_mask):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if X_count > best_size:
                best_size = X_count
                best_set_mask = X_mask
            return
            
        # Get vertices in P
        verts = []
        temp = P_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            verts.append(v)
            temp &= temp - 1
            
        if not verts:
            return
            
        # Sort by degree in P
        def deg_in_P(v):
            return popcount(graph_mask[v] & P_mask)
            
        verts.sort(key=deg_in_P, reverse=True)
        
        # Greedy Coloring
        colors = {}
        max_color = 0
        
        for v in verts:
            used = set()
            neigh_mask = graph_mask[v] & P_mask
            temp_neigh = neigh_mask
            while temp_neigh:
                u = (temp_neigh & -temp_neigh).bit_length() - 1
                if u in colors:
                    used.add(colors[u])
                temp_neigh &= temp_neigh - 1
                
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        if X_count + max_color <= best_size:
            return
            
        # Branch
        branch_order = sorted(verts, key=lambda x: colors[x], reverse=True)
        
        for v in branch_order:
            if X_count + colors[v] <= best_size:
                break
                
            # Include v
            new_P1 = P_mask & ~(graph_mask[v] | (1 << v))
            new_X_mask = X_mask | (1 << v)
            
            expand(new_P1, X_count + 1, new_X_mask)
            
            if time.time() > deadline:
                return
            
            # Exclude v
            P_mask &= ~(1 << v)
            
    # Start B&B
    expand(initial_P, 0, 0)
    
    # Reconstruct set from best_set_mask
    if best_set_mask == 0:
        return [1]
        
    result = []
    for i in range(1, n + 1):
        if best_set_mask & (1 << i):
            result.append(i)
            
    return result

# Run once to verify? No, construct is called by evaluator.
# Just ensure it returns a list.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
    # print(res)
