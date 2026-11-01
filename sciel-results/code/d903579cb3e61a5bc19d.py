import numpy as np
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Parameters
    q = k
    # d such that girth >= d+5
    # g=16 -> d=11
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # GF(q) operations
    # For q=3, simple mod 3
    # We represent vertices as tuples (side, x_0, ..., x_d)
    # side 0: point, side 1: line
    
    # We will use BFS.
    # Level 0: start vertex
    # We store vertices in lists.
    
    # For q=3, we can use integers 0, 1, 2.
    
    # Rule generation
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # BFS
    # Vertex: (side, x) where x is tuple of d+1 ints (x[0] is dummy/None in baseline, but we can just use d+1)
    # Baseline uses (None, ...) but for numpy we use a fixed size array.
    # Let's use shape (d+1).
    
    start_x = np.zeros(d+1, dtype=np.int8)
    start_v = (0, tuple(start_x))
    
    index = {start_v: 0}
    order = [start_v]
    edges = []
    
    current_level = [0] # indices in order
    
    level = 0
    while current_level:
        if time.time() > deadline:
            break
        
        next_level = []
        # Vectorize neighbor generation for current_level
        # Extract x arrays
        # x_side, x_vals
        # For each vertex in current_level, generate q neighbors on the other side.
        
        # This is tricky to vectorize fully because the output is a list of tuples.
        # We can compute the new x values for all vertices in current_level at once.
        
        side, x_vals = zip(*[order[i] for i in current_level])
        side_arr = np.array(side, dtype=np.int8)
        x_arr = np.array(x_vals, dtype=np.int8) # shape (L, d+1)
        
        # For each of the q 'first' coordinates
        for first in range(q):
            # y[0] is dummy, y[1] = first
            # y[c] for c in 2..d
            y_arr = np.zeros_like(x_arr)
            y_arr[:, 1] = first
            
            for c in range(2, d + 1):
                a, b = rule[c]
                # x_arr indices: 0 is dummy, 1 is x[1] (which is the first coord in points?)
                # Wait, the baseline x is (None, x_1, ..., x_d).
                # So x_arr[:, i] corresponds to x_i.
                
                # point p ~ line l
                # l[c] - p[c] = l[a] * p[b]
                # y[c] = x[c] + y[a] * x[b]  (if side 0 -> point to line)
                # y[c] = x[c] - x[a] * y[b]  (if side 1 -> line to point)
                
                # We have x_arr for the current vertices.
                # We need to compute y_arr.
                # y[a] and y[b] are already computed for c > a, b?
                # Yes, we iterate c from 2 to d.
                
                # Mask for side 0 and 1
                mask0 = (side_arr == 0)
                mask1 = (side_arr == 1)
                
                # Compute term
                term = np.zeros_like(x_arr[:, c])
                
                if np.any(mask0):
                    # y[a] * x[b]
                    # y[a] is available. x[b] is from current vertex.
                    term[mask0] = (y_arr[mask0, a] * x_arr[mask0, b]) % q
                    
                if np.any(mask1):
                    # x[a] * y[b]
                    term[mask1] = (x_arr[mask1, a] * y_arr[mask1, b]) % q
                
                if np.any(mask0):
                    y_arr[mask0, c] = (x_arr[mask0, c] + term[mask0]) % q
                if np.any(mask1):
                    y_arr[mask1, c] = (x_arr[mask1, c] - term[mask1]) % q

            # Now y_arr is computed for all vertices in current_level for this 'first'.
            # New vertices are (1-side, y_tuple)
            
            # Convert to tuples and check index
            # This part is hard to vectorize (dict lookup).
            # We can batch the lookup?
            
            new_side = 1 - side_arr
            # y_arr has shape (L, d+1)
            # We need to add these to index if not present.
            
            # We can use a set of tuples for fast lookup.
            # But converting numpy array to list of tuples is slow.
            # We can use a hash map from bytes to index?
            # For q=3, d=11, size is small.
            
            # Fallback to Python loop for the lookup, but the heavy math is done.
            for i in range(len(current_level)):
                v_idx = current_level[i]
                new_v = (int(new_side[i]), tuple(y_arr[i]))
                if new_v not in index:
                    index[new_v] = len(order)
                    order.append(new_v)
                    next_level.append(index[new_v])
                if side_arr[i] == 0:
                    edges.append((v_idx, index[new_v]))

        current_level = next_level
        level += 1
        if level > 50: # Safety
            break

    return len(order), edges
