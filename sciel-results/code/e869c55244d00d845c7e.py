        def find_clique_bitset(cand_list, n, deadline):
            N = len(cand_list)
            if N < n: return None
            
            # Precompute conflict masks
            # conflict_mask[i] is a bitset of indices j (j > i) that conflict with i?
            # Or just j != i.
            # Let's compute full conflict matrix as bitsets.
            # conflict[i] has bit j set if cand_list[i] and cand_list[j] conflict.
            
            conflicts = [0] * N
            for i in range(N):
                mi = cand_list[i][1]
                for j in range(i + 1, N):
                    mj = cand_list[j][1]
                    if mi & mj:
                        conflicts[i] |= (1 << j)
                        conflicts[j] |= (1 << i)
                        
            # Search
            def search(available, count, current_clique):
                if time.time() > deadline: return None
                if count == n: return current_clique
                
                # Pick a pivot?
                # Just pick the lowest set bit?
                # To enforce symmetry, we should pick from available.
                # The available set already has lower indices masked out?
                # No, we mask them out *after* picking.
                
                # Get list of available indices
                # It's faster to iterate set bits.
                
                # Heuristic: Pick the index with the fewest neighbors in available?
                # Or just lowest index?
                
                # Let's try lowest index first.
                # Find lowest set bit in available
                if available == 0: return None
                
                # Iterate through set bits
                # We can use a while loop
                avail = available
                while avail:
                    # Get lowest set bit
                    lsb = avail & (-avail)
                    i = lsb.bit_length() - 1
                    
                    # Try i
                    # New available: available without i and without conflicts[i]
                    new_avail = available & ~(lsb | conflicts[i])
                    
                    # Mask out indices < i to maintain order?
                    # Actually, if we always pick from `available`, and `available` initially contains all,
                    # and we remove conflicts, we might pick i=5 then i=2.
                    # To prevent permutations, we should only allow indices > last_picked.
                    # So, when we pick i, the next available should be `new_avail & ~((1 << (i+1)) - 1)`.
                    
                    next_avail = new_avail & ~((1 << (i + 1)) - 1)
                    
                    res = search(next_avail, count + 1, current_clique + [cand_list[i][2]])
                    if res: return res
                    
                    avail ^= lsb # Remove i from iteration
            
            return search((1 << N) - 1, 0, [])
        