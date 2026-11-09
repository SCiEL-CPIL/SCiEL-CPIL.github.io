<<<<<<< SEARCH
    def is_valid_incremental(S, x):
        # S is valid.
        # Generate sums of S
        sums_S = {0}
        for val in S:
            new_sums = set()
            for s in sums_S:
                new_sums.add(s + val)
            sums_S = sums_S | new_sums
        
        # New sums are x + s for s in sums_S
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness
        # Intersection of sums_S and new_sums must be empty
        if sums_S & new_sums:
            return False
            
        # Check divisibility
        # All pairs in (sums_S U new_sums) \ {0}
        # We know sums_S \ {0} is antichain.
        # We need to check:
        # - new_sums \ {0} is antichain
        # - cross pairs
        
        non_empty_new = [s for s in new_sums if s != 0]
        non_empty_old = [s for s in sums_S if s != 0]
        
        # Check internal new
        n = len(non_empty_new)
        for i in range(n):
            u = non_empty_new[i]
            for j in range(n):
                if i == j: continue
                v = non_empty_new[j]
                if v % u == 0 or u % v == 0:
                    return False
        
        # Check cross
        for u in non_empty_new:
            for v in non_empty_old:
                if v % u == 0 or u % v == 0:
                    return False
                    
        return True
=======
    def get_sums(S):
        """Compute all subset sums including 0."""
        sums = {0}
        for val in S:
            new_sums = {s + val for s in sums}
            sums = sums | new_sums
        return sums
    
    def is_valid_incremental(S, x):
        # S is valid.
        sums_S = get_sums(S)
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness
        if sums_S & new_sums:
            return False
            
        non_empty_new = [s for s in new_sums if s != 0]
        non_empty_old = [s for s in sums_S if s != 0]
        
        # Check internal new
        n = len(non_empty_new)
        for i in range(n):
            u = non_empty_new[i]
            for j in range(n):
                if i == j: continue
                if u < v or v < u:
                    small, large = (u, v) if u < v else (v, u)
                    if large % small == 0:
                        return False
                else:
                    # u == v already caught by distinctness
                    pass
        
        # Check cross
        for u in non_empty_new:
            for v in non_empty_old:
                if u < v:
                    if v % u == 0:
                        return False
                elif v < u:
                    if u % v == 0:
                        return False
        
        return True
>>>>>>> REPLACE

