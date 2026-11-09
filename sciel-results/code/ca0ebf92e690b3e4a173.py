<<<<<<< SEARCH
    # 2. Branch and Bound for Maximum Independent Set
    # We will use a recursive search.
    # State: (C, current_size, current_set)
    # C: bitset of available vertices
    # We branch on a vertex v in C.
    # Include v: new C = C & ~conflicts[v] & ~vertex_bit[v]
    # Exclude v: new C = C & ~vertex_bit[v]
    
    # To make it efficient, we need good pruning.
    # Bound: current_size + upper_bound(C) <= best_size -> prune
    # Upper bound on independent set size in C:
    # Simple bound: |C|
    # Better bound: Greedy coloring of the conflict graph induced by C?
    # Or just |C| is often enough if we order vertices well.
    
    # Let's implement a simple B&B with |C| bound first, and see if it's fast enough.
    # If not, we can add a greedy coloring bound.
    
    # To improve pruning, we can order the branching vertices.
    # Pick a vertex v in C.
    # Strategy: Pick v with max degree in conflict graph induced by C?
    # This tends to reduce the "include" branch significantly.
    
    def expand(C, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return False
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return True
            
        # Pruning:
        if current_size + C.bit_count() <= best_size:
            return True
            
        # Select vertex to branch on
        # Heuristic: Max degree in conflict graph induced by C
        best_v = -1
        max_deg = -1
        temp = C
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (conflicts[v] & C).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return True
            
        v_bit = vertex_bit[best_v]
        
        # Branch 1: Include best_v
        # New C: remove best_v and its neighbors
        new_C_include = C & ~(conflicts[best_v] | v_bit)
        current_set.append(best_v)
        
        # Check time
        if time.time() > deadline:
            current_set.pop()
            return False
            
        expand(new_C_include, current_size + 1, current_set)
        current_set.pop()
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude best_v
        new_C_exclude = C & ~v_bit
        expand(new_C_exclude, current_size, current_set)
        
        return True
=======
    # 2. Branch and Bound for Maximum Independent Set
    # We will use a recursive search with greedy coloring upper bound.
    
    def greedy_color_bound(C):
        """
        Compute an upper bound on the size of an independent set in the
        conflict graph induced by C using greedy coloring.
        Returns the number of colors used, which is an upper bound on
        the independence number.
        """
        if C == 0:
            return 0
        
        # Greedy coloring: assign colors to vertices in order of decreasing degree
        # in the conflict graph induced by C
        remaining = C
        color_count = 0
        
        while remaining:
            color_count += 1
            
            # Find a maximal independent set in the remaining graph
            # (this becomes one color class)
            color_class = 0
            temp = remaining
            while temp:
                # Pick vertex with max degree in remaining
                best_v = -1
                max_deg = -1
                t = temp
                while t:
                    lsb = t & -t
                    v = lsb.bit_length() - 1
                    deg = (conflicts[v] & remaining).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                    t ^= lsb
                
                if best_v == -1:
                    break
                
                # Add best_v to color class
                color_class |= vertex_bit[best_v]
                # Remove best_v and its neighbors from remaining
                remaining &= ~(conflicts[best_v] | vertex_bit[best_v])
            
            # Remove the entire color class from remaining
            remaining &= ~color_class
        
        return color_count
    
    def expand(C, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return False
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return True
            
        # Pruning with greedy coloring bound
        upper_bound = greedy_color_bound(C)
        if current_size + upper_bound <= best_size:
            return True
            
        # Select vertex to branch on
        # Heuristic: Max degree in conflict graph induced by C
        best_v = -1
        max_deg = -1
        temp = C
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (conflicts[v] & C).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return True
            
        v_bit = vertex_bit[best_v]
        
        # Branch 1: Include best_v
        # New C: remove best_v and its neighbors
        new_C_include = C & ~(conflicts[best_v] | v_bit)
        current_set.append(best_v)
        
        # Check time
        if time.time() > deadline:
            current_set.pop()
            return False
            
        expand(new_C_include, current_size + 1, current_set)
        current_set.pop()
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude best_v
        new_C_exclude = C & ~v_bit
        expand(new_C_exclude, current_size, current_set)
        
        return True
>>>>>>> REPLACE

