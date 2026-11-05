import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    # 1. Build Compatibility Graph
    # neighbors[i] is a bitmask of j such that lcm(i, j) <= N
    neighbors = [0] * (N + 1)
    
    # Precompute LCM conflicts or compatibilities
    # For each pair (i, j), check lcm(i, j) <= N
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = i * j // g
            if l <= N:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)

    # 2. Vertex Ordering
    # Sort vertices by degree (number of compatible neighbors) descending
    # This helps in greedy coloring bound
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: bin(neighbors[x]).count('1'), reverse=True)
    
    # Map original vertex index to position in sorted list? 
    # Actually, for Tomita's algorithm, we can just work with the sorted list 
    # and map back to original indices for the result, or keep original indices 
    # and use the sorted order for iteration.
    
    # Let's keep original indices in the bitmasks, but iterate in sorted order.
    
    def greedy_color(P, order):
        """
        P: bitmask of candidates
        order: list of vertices in sorted order (to iterate P)
        Returns:
        colors: dict vertex -> color number
        color_class: list of bitmasks, color_class[c] contains vertices with color c
        max_color: max color used + 1 (total colors)
        """
        colors = {}
        color_class = [0]
        max_color = 0
        
        # Iterate through vertices in P in the given order
        # We need to extract vertices from P. 
        # Instead of iterating all N, we can iterate the sorted list and check if in P.
        # Or, we can just iterate through the sorted list, and if v is in P, color it.
        
        for v in order:
            if not (P >> v) & 1:
                continue
                
            # Find smallest color not used by neighbors of v that are already colored
            # Neighbors of v in P: N_v = neighbors[v] & P
            # But we only care about already colored neighbors.
            
            # Efficient way:
            # For each color c, check if any neighbor of v has color c.
            # We can maintain a bitmask for each color class.
            # If (neighbors[v] & color_class[c]) != 0, then color c is used.
            
            used_colors = set()
            # Optimization: We can just check against the color_class masks
            for c in range(max_color):
                if (neighbors[v] & color_class[c]) != 0:
                    used_colors.add(c)
            
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c == max_color:
                color_class.append(0)
                max_color += 1
            color_class[c] |= (1 << v)
            
        return colors, color_class, max_color

    best_clique = []
    
    def expand(R, P):
        nonlocal best_clique
        
        if time.time() - start_time > time_limit:
            return
            
        if P == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Upper bound: len(R) + popcount(P)
        # If this is <= len(best_clique), prune
        # popcount for Python int: bin(P).count('1')
        if len(R) + bin(P).count('1') <= len(best_clique):
            return
            
        # Greedy Coloring Bound
        colors, color_class, max_color = greedy_color(P, vertices)
        
        if len(R) + max_color <= len(best_clique):
            return
            
        # Process vertices in reverse order of color
        # We want to process vertices with the highest color first
        # Create a list of vertices in P sorted by color descending
        # Then by degree? 
        # Standard Tomita: sort by color desc.
        
        # We have colors dict.
        # Let's create a list of (color, vertex) for v in P
        # Then sort by color descending.
        
        # To avoid overhead, we can just iterate color classes from high to low
        # and process vertices in each class.
        
        # However, the standard algorithm iterates vertices one by one.
        # Let's construct the ordered list.
        p_vertices = [v for v in vertices if (P >> v) & 1]
        p_vertices.sort(key=lambda v: colors.get(v, 0), reverse=True)
        
        for v in p_vertices:
            if not (P >> v) & 1:
                continue
                
            # Include v in clique
            R.append(v)
            
            # New candidates are neighbors of v that are in P
            new_P = P & neighbors[v]
            
            expand(R, new_P)
            
            R.pop()
            
            # Remove v from P for subsequent iterations
            P &= ~(1 << v)
            
            # Re-evaluate bound with remaining P
            # Simple bound
            if len(R) + bin(P).count('1') <= len(best_clique):
                break

    # Initial call
    # R is empty, P is all vertices
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << i)
        
    expand([], initial_P)
    
    # Return the best clique found
    return sorted(best_clique)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
