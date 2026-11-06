import math
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin

    n = 256
    max_sum = 2 * n
    
    # 1. Identify squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # 2. Filter candidates: a is valid if 2*a is not a square
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
            
    m = len(candidates)
    
    # Map original values to indices 0..m-1
    val_to_idx = {val: i for i, val in enumerate(candidates)}
    
    # 3. Build Conflict Graph G (edges if sum is square)
    # We will build the Complement Graph H (edges if sum is NOT square)
    # H[i] is a bitmask of vertices j such that candidates[i] + candidates[j] is NOT a square
    # Note: H should not have self-loops.
    
    H = [0] * m
    for i in range(m):
        vi = candidates[i]
        mask_i = 0
        for j in range(m):
            if i == j:
                continue
            vj = candidates[j]
            if (vi + vj) not in squares:
                mask_i |= (1 << j)
        H[i] = mask_i
        
    # 4. Maximum Clique in H is Maximum Independent Set in G
    # Use Branch and Bound with Greedy Coloring Bound
    
    best_clique = 0
    best_vertices = []
    
    # Order vertices by degree in H (descending) to help pruning
    # We will reorder the vertex indices to improve the search
    # Let's keep the original indices but sort the candidates list? 
    # It's easier to just work with the current indices and use a good ordering in the recursion.
    
    # Greedy Coloring of a subgraph defined by a bitmask P
    # Returns the number of colors used.
    def greedy_color_bound(P):
        # P is a bitmask of vertices
        if P == 0:
            return 0
        colors = 0
        remaining = P
        while remaining:
            colors += 1
            # Greedily assign this color to a maximal independent set in H[remaining]
            # Which is a maximal clique in G[remaining]? No.
            # We are coloring H. We want to find a large independent set in H to remove?
            # No, greedy coloring for bound:
            # We want to partition V into independent sets of H.
            # An independent set in H is a clique in G.
            # Wait, the bound is: Max Clique in H <= Number of Colors.
            # Standard greedy coloring for Max Clique bound:
            # Order vertices. Assign smallest available color.
            # The max color assigned is the bound.
            
            # Let's implement a simple greedy coloring that returns the max color.
            # To make it a good bound, we should process vertices in a specific order.
            # A common heuristic is to order by degree in the remaining subgraph.
            
            # However, for speed, a simple "first fit" on a fixed order (e.g., index) might be weak.
            # Let's use the "DSATUR" like heuristic or just simple sequential coloring on the bits.
            # Given the small N (256), we can do a slightly smarter greedy.
            
            # Actually, the standard McKay bound uses a coloring where we try to pack as many vertices as possible into each color class.
            # This is equivalent to finding a maximal independent set in H (which is a maximal clique in G).
            # Let's find a maximal independent set in H[remaining].
            
            color_class = 0
            temp_remaining = remaining
            while temp_remaining:
                # Pick a vertex v from temp_remaining
                # To maximize the color class, we should pick vertices that conflict with few others in the color class?
                # Or just pick the first one?
                # Let's pick the vertex with the lowest index for determinism and speed.
                v_idx = (temp_remaining & -temp_remaining).bit_length() - 1
                v_bit = 1 << v_idx
                
                # Add v to color class
                color_class |= v_bit
                
                # Remove v and all its neighbors in H from temp_remaining
                # Because they can't be in the same independent set in H.
                # Neighbors in H are H[v_idx].
                temp_remaining &= ~(H[v_idx] | v_bit)
                
            # Remove color_class from remaining
            remaining &= ~color_class
            
        return colors

    # Recursive B&B for Max Clique in H
    def expand(P, X, depth, current_clique):
        nonlocal best_clique, best_vertices
        
        if P == 0:
            if depth > best_clique:
                best_clique = depth
                # Reconstruct vertices from current_clique bitmask
                best_vertices = [i for i in range(m) if (current_clique >> i) & 1]
            return
        
        # Bound
        bound = greedy_color_bound(P)
        if depth + bound <= best_clique:
            return
            
        # Order vertices in P
        # We want to pick vertices that are likely to lead to large cliques.
        # Usually, vertices with high degree in the remaining subgraph are good candidates.
        # Or, we can just iterate through the bits.
        
        # To improve pruning, we can order the vertices in P.
        # A simple heuristic: sort by degree in P (descending).
        # Calculating degree for all vertices in P every step is O(N^2).
        # Given N=256, this is acceptable.
        
        # Let's collect vertices in P
        verts = []
        temp = P
        while temp:
            v_idx = (temp & -temp).bit_length() - 1
            verts.append(v_idx)
            temp &= temp - 1
            
        # Calculate degrees in the subgraph induced by P
        # deg(v) = popcount(H[v] & P)
        # Sort by degree descending
        verts.sort(key=lambda v: bin(H[v] & P).count('1'), reverse=True)
        
        for v in verts:
            if time.time() - start_time > TIME_LIMIT:
                return
            
            v_bit = 1 << v
            if not (P & v_bit):
                continue # Already processed or not in P?
            
            # Include v
            new_P = P & H[v]
            new_clique = current_clique | v_bit
            
            expand(new_P, X, depth + 1, new_clique)
            
            # Exclude v
            P &= ~v_bit
            X |= v_bit # Optional
            
            # Check time again?
            if time.time() - start_time > TIME_LIMIT:
                return
                
    # Initial call
    # We can start with all vertices
    # To help, we can use a greedy initial solution to set best_clique high
    # Greedy Max Clique in H:
    current_clique = 0
    P = (1 << m) - 1
    while P:
        # Pick vertex with max degree in P
        best_v = -1
        max_deg = -1
        temp = P
        while temp:
            v_idx = (temp & -temp).bit_length() - 1
            deg = bin(H[v_idx] & P).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v_idx
            temp &= temp - 1
            
        if best_v == -1:
            break
            
        current_clique |= (1 << best_v)
        P &= H[best_v]
        
    best_clique = bin(current_clique).count('1')
    best_vertices = [i for i in range(m) if (current_clique >> i) & 1]
    
    # Run B&B
    # We pass P as all vertices
    expand((1 << m) - 1, 0, 0, 0)
    
    # Convert indices back to values
    result = [candidates[i] for i in best_vertices]
    
    # Sort for consistent output
    result.sort()
    
    # Verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
