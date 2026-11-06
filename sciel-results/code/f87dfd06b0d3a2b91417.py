    def max_independent_set_size_bitset(n, adj_bits):
        """
        adj_bits: list of integers, where adj_bits[i] has bit j set if edge (i,j) exists.
        """
        best = [0]
        
        def branch(current_size, candidates_mask):
            # Upper bound: current_size + popcount(candidates_mask)
            # Python int.bit_count() is fast
            if current_size + candidates_mask.bit_count() <= best[0]:
                return
            
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with highest degree in the induced subgraph
            # To get degree of v in candidates:
            # deg(v) = (adj_bits[v] & candidates_mask).bit_count()
            
            # Find v with max degree
            max_deg = -1
            v = -1
            temp_mask = candidates_mask
            while temp_mask:
                # Get lowest set bit
                lb = temp_mask & -temp_mask
                idx = lb.bit_length() - 1
                deg = (adj_bits[idx] & candidates_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp_mask ^= lb
                
            if v == -1:
                return
                
            # Branch 1: Include v
            # Remove v and its neighbors from candidates
            new_mask = candidates_mask & ~adj_bits[v] & ~(1 << v)
            branch(current_size + 1, new_mask)
            
            # Branch 2: Exclude v
            branch(current_size, candidates_mask & ~(1 << v))
            
        branch(0, (1 << n) - 1)
        return best[0]
