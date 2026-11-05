import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # We will attempt to find a valid coloring for n=30.
    # If that fails, we fall back to n=29.
    
    def check_violation(n, colors):
        """
        Returns None if valid.
        Returns a list of 6 vertices forming a monochromatic K2,2,2 if invalid.
        colors is a list of 0/1 for pairs (i,j).
        """
        # Build adjacency bitmasks for color 0 and 1
        # mask_c[i] is a bitmask of vertices j such that edge (i,j) is color c
        mask0 = [0] * n
        mask1 = [0] * n
        
        # Helper to get index
        def idx(i, j):
            if i > j: i, j = j, i
            return i * (i - 1) // 2 + j - i - 1
            # Wait, standard lexicographic index for i<j:
            # (0,1), (0,2), ..., (0,n-1), (1,2), ...
            # Index for (i,j) with i<j:
            # Sum of k for k=0 to i-1 of (n-1-k) + (j-i-1)
            # Or simpler:
            # total_edges = n*(n-1)//2
            # We can just use a 2D array or a dict for speed in Python, 
            # but bitmasks are better for intersection.
            # Let's build masks from the list.
        
        # Rebuild masks efficiently
        # Iterate through all pairs
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    mask0[i] |= (1 << j)
                    mask0[j] |= (1 << i)
                else:
                    mask1[i] |= (1 << j)
                    mask1[j] |= (1 << i)
                k += 1
        
        # Check for K2,2,2 in color 0
        # Iterate over pairs (a,b) and (c,d)
        # To avoid duplicates and ensure disjointness, we can iterate 4-tuples
        # This is O(n^4). For n=30, ~800k iterations.
        # Each iteration involves bitmask ANDs.
        
        for a in range(n):
            for b in range(a + 1, n):
                # Check if edge (a,b) is color 0
                if not (mask0[a] & (1 << b)):
                    continue
                
                for c in range(b + 1, n):
                    # Check if edge (c,d) is disjoint from a,b and color 0
                    # We need d > c
                    # Also need c,d not in {a,b}
                    # Since c > b >= a, c is distinct from a,b.
                    # We need d such that d is not a or b.
                    # Since d > c > b, d is distinct.
                    
                    # Check if (c,d) is color 0
                    # We iterate d
                    for d in range(c + 1, n):
                        if not (mask0[c] & (1 << d)):
                            continue
                        
                        # We have edges (a,b) and (c,d) of color 0.
                        # Check if they form a K2,2:
                        # Need (a,c), (a,d), (b,c), (b,d) to be color 0.
                        if (mask0[a] & (1 << c)) and (mask0[a] & (1 << d)) and \
                           (mask0[b] & (1 << c)) and (mask0[b] & (1 << d)):
                            
                            # We have a K2,2 on {a,b,c,d} in color 0.
                            # Find common neighbors of a,b,c,d in color 0.
                            common = mask0[a] & mask0[b] & mask0[c] & mask0[d]
                            # Remove a,b,c,d from common (they are not neighbors of themselves)
                            # Actually, mask0[i] does not have bit i set.
                            # But a is in mask0[b], etc.
                            # We need vertices e, f such that e,f are in common.
                            # And e,f must be distinct from a,b,c,d.
                            # Since mask0[x] doesn't contain x, 
                            # and a,b,c,d are not neighbors of themselves,
                            # common does not contain a,b,c,d?
                            # Wait. mask0[a] contains b,c,d if edges exist.
                            # So common contains a? No, mask0[b] contains a.
                            # mask0[a] & mask0[b] contains c,d if they are common neighbors.
                            # Does it contain a? No, mask0[a] doesn't have bit a.
                            # So common is a subset of V \ {a,b,c,d}.
                            
                            # We need at least 2 bits set in common.
                            if common & (common - 1): # Check if at least 2 bits set
                                # Extract two vertices
                                # Get lowest set bit
                                e_bit = common & -common
                                e = e_bit.bit_length() - 1
                                common2 = common ^ e_bit
                                f_bit = common2 & -common2
                                f = f_bit.bit_length() - 1
                                return [a, b, c, d, e, f]
        
        # Check for K2,2,2 in color 1
        for a in range(n):
            for b in range(a + 1, n):
                if not (mask1[a] & (1 << b)):
                    continue
                
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not (mask1[c] & (1 << d)):
                            continue
                        
                        if (mask1[a] & (1 << c)) and (mask1[a] & (1 << d)) and \
                           (mask1[b] & (1 << c)) and (mask1[b] & (1 << d)):
                            
                            common = mask1[a] & mask1[b] & mask1[c] & mask1[d]
                            if common & (common - 1):
                                e_bit = common & -common
                                e = e_bit.bit_length() - 1
                                common2 = common ^ e_bit
                                f_bit = common2 & -common2
                                f = f_bit.bit_length() - 1
                                return [a, b, c, d, e, f]
        
        return None

    def try_find_coloring(n):
        # Initialize random
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Local search
        # We will run for a limited number of steps or time
        steps = 0
        max_steps = 5000
        last_check = time.time()
        
        while steps < max_steps:
            if time.time() > deadline - 5:
                break
                
            violation = check_violation(n, colors)
            if violation is None:
                return colors
            
            # Flip an edge in the violation
            # The violation is a set of 6 vertices.
            # We can pick a random edge among the 15 possible edges in K6?
            # No, only the edges present in the K2,2,2 structure matter for the violation.
            # The structure is defined by 3 pairs.
            # Let's just pick a random pair from the 6 vertices and flip it if it's part of the monochromatic structure.
            # Actually, any edge flip that breaks the monochromatic connectivity helps.
            # Let's pick a random edge from the complete graph on the 6 vertices.
            v = violation
            random.shuffle(v)
            i, j = v[0], v[1]
            if i > j: i, j = j, i
            # Calculate index
            # Index formula for (i,j) with i<j:
            # k = i*(n-1) - i*(i+1)//2 + (j-i-1) ?
            # Let's use a helper or precomputed map for speed if we do this often.
            # For n=30, num_edges=435. Simple loop is fine.
            idx = 0
            for x in range(n):
                for y in range(x + 1, n):
                    if x == i and y == j:
                        break
                    idx += 1
                else:
                    continue
                break
            
            colors[idx] = 1 - colors[idx]
            steps += 1
            
        return None # Failed to find valid coloring within limits

    # Try n=30
    result = try_find_coloring(30)
    if result:
        return {'n': 30, 'colors': result}
        
    # Try n=29
    if time.time() < deadline - 10:
        result = try_find_coloring(29)
        if result:
            return {'n': 29, 'colors': result}
            
    # Fallback to known good n=17 (Paley)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
