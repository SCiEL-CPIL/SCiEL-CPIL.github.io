"""
Rewritten program using Lazebnik-Ustimenko-Woldar (LUW) construction.
This implementation uses a direct BFS with explicit state tracking to 
construct the q-regular graph of girth at least d+5.
"""
# EVOLVE-BLOCK-START
def construct(k, g):
    import collections
    
    # Determine parameters for the LUW construction CD(d, q)
    # We need girth >= g.
    # For odd d, girth >= d + 5.
    # So we need d + 5 >= g => d >= g - 5.
    # Also d must be odd for the standard bound, or we can check even d bounds.
    # Let's select the smallest odd d >= max(3, g - 5).
    
    q = k
    min_d = g - 5
    if min_d < 3:
        min_d = 3
        
    d = min_d
    if d % 2 == 0:
        d += 1
        
    # If q=4, use GF(4) arithmetic, otherwise use Z_q (assuming q is prime)
    if q == 4:
        # GF(4) addition is XOR
        # GF(4) multiplication table
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def gf_add(x, y):
            return x ^ y
        def gf_sub(x, y):
            return x ^ y
        def gf_mul(x, y):
            return mul_table[x][y]
    else:
        def gf_add(x, y):
            return (x + y) % q
        def gf_sub(x, y):
            return (x - y) % q
        def gf_mul(x, y):
            return (x * y) % q

    # Precompute the rules for indices a, b for each column c in 2..d
    # The rule pattern cycles every 4 in the LUW construction.
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    
    for c in range(5, d + 1):
        # Pattern repeats every 4: (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        # where c0 is the start of the cycle block
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rules[c] = (1, c0 - 2)
        elif r == 1:
            rules[c] = (c0 - 1, 1)
        elif r == 2:
            rules[c] = (c0, 1)
        else:
            rules[c] = (1, c0 + 1)

    # Start state: side 0 (point), vector of zeros of length d
    # We represent a vertex as (side, vector_tuple)
    start_state = (0, tuple([0] * d))
    
    # BFS to find connected component and count vertices/edges
    # Since the graph is regular and connected (usually), we can just BFS from start.
    # We need to return the full graph if it's connected, or the component.
    # LUW graphs are connected for prime q.
    
    visited = {start_state: 0}
    queue = collections.deque([start_state])
    vertex_id = 0
    edges = []
    
    while queue:
        current = queue.popleft()
        side, vec = current
        
        # Generate neighbors
        # The neighbor is on the other side (1 - side)
        # The first element of the new vector is free (iterates 0..q-1)
        # The rest are determined by the recurrence.
        
        for first_val in range(q):
            # Initialize new vector
            # vec indices are 0..d-1 corresponding to columns 1..d
            # In the formula, y[c] depends on y[a] and x[b]
            # Note: The original code used 1-based indexing for the vector in the formula
            # Here we use 0-based indexing for the Python tuple.
            # Formula: y[c] - x[c] = y[a] * x[b]  (for side 0 -> 1)
            #           x[c] - y[c] = x[a] * y[b]  (for side 1 -> 0)
            
            new_vec_list = [0] * d
            new_vec_list[0] = first_val # Corresponds to column 1
            
            if side == 0:
                # Point to Line
                # y[c] = x[c] + y[a] * x[b]
                for c_idx in range(1, d): # c from 2 to d
                    c_val = c_idx + 1
                    a, b = rules[c_val]
                    # a and b are 1-based indices in the original formula
                    # Convert to 0-based for Python list
                    # x is the current vector (point)
                    # y is the new vector (line)
                    # y[a-1] * x[b-1]
                    val = gf_add(vec[c_idx], gf_mul(new_vec_list[a-1], vec[b-1]))
                    new_vec_list[c_idx] = val
            else:
                # Line to Point
                # x[c] = y[c] + x[a] * y[b]
                # Here current is 'y' (line), new is 'x' (point)
                for c_idx in range(1, d):
                    c_val = c_idx + 1
                    a, b = rules[c_val]
                    val = gf_add(vec[c_idx], gf_mul(vec[a-1], new_vec_list[b-1]))
                    new_vec_list[c_idx] = val
            
            neighbor_state = (1 - side, tuple(new_vec_list))
            
            if neighbor_state not in visited:
                visited[neighbor_state] = len(visited)
                queue.append(neighbor_state)
                
            # Add edge
            # To avoid duplicate edges, only add if current side is 0
            if side == 0:
                edges.append((visited[current], visited[neighbor_state]))
                
    n = len(visited)
    return n, edges

# EVOLVE-BLOCK-END
