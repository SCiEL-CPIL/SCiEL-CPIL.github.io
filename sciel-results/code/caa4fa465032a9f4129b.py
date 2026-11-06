    def min_triangle_transversal(triangles):
        if not triangles:
            return 0
        
        # Convert to list for sorting
        tri_list = list(triangles)
        # Sort by popcount to ensure consistent ordering for memoization keys
        # and to pick the most constrained triangle first
        tri_list.sort(key=lambda x: bin(x).count('1'))
        
        memo = {}
        
        def bnb(active):
            if not active:
                return 0
            
            # Create a key for memoization
            # active is a list of ints. We sort it to make the key canonical.
            # Since we always branch on the min-popcount triangle, the order of the rest
            # in the list might vary depending on how we filtered them.
            # So we must sort for the key.
            key = tuple(sorted(active))
            
            if key in memo:
                return memo[key]
            
            # Pick the triangle with the smallest popcount
            # Since active is sorted by popcount in the initial list?
            # No, after filtering, the relative order is preserved from the sorted list,
            # but some are removed. So the first element of the sorted list is still the min popcount?
            # Yes, if we sorted the initial list, and we just filter, the min popcount element
            # in the remaining set is the first one in the sorted list that is present.
            # However, `active` in the recursive call is a list derived from `active` in the parent.
            # If we maintain the sorted order, `active[0]` is the min popcount.
            
            # But wait, the `key` uses `sorted(active)`.
            # So `active` itself doesn't need to be sorted for the logic, but for efficiency
            # we can just use `min(active, key=...)`.
            
            min_tri = min(active, key=lambda x: bin(x).count('1'))
            
            best_cost = 18 # n
            
            temp = min_tri
            while temp:
                v_bit = temp & -temp
                
                # Filter active triangles: keep those that do NOT contain v_bit
                new_active = [t for t in active if not (t & v_bit)]
                
                cost = bnb(new_active)
                if cost < best_cost:
                    best_cost = cost
                    if best_cost == 1:
                        break # Can't do better than 1 for non-empty active
                
                temp &= temp - 1
            
            memo[key] = best_cost
            return best_cost

        return bnb(tri_list)
