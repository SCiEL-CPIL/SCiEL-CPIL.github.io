<<<<<<< SEARCH
    def is_dissociated(S):
        """
        Check if all subset sums of reciprocals of S are distinct.
        S is a list of integers.
        """
        if not S:
            return True
        
        n = len(S)
        # If n is large, 2^n is large.
        # For n=20, 2^20 = 1,048,576.
        # Generating 1M Fractions might be slow.
        # Let's optimize.
        
        sums = [Fraction(0, 1)]
        for x in S:
            val = reciprocals[x]
            # Extend sums list
            # Optimization: use list comprehension or loop
            # new_sums = [s + val for s in sums]
            # sums.extend(new_sums)
            
            # To avoid creating a new list every time, we can do:
            current_len = len(sums)
            for i in range(current_len):
                sums.append(sums[i] + val)
                
        # Check for duplicates
        # Using a set is O(2^n).
        # If there are duplicates, len(set) < len(sums)
        if len(sums) != len(set(sums)):
            return False
        return True

    def find_collision(S):
        """
        Find a non-trivial linear combination of reciprocals of S that sums to 0.
        Returns a list of (coefficient, element) pairs, or None if dissociated.
        Coefficients are in {-1, 0, 1}.
        """
        n = len(S)
        sums = [Fraction(0, 1)]
        masks = [0]
        
        for i, x in enumerate(S):
            val = reciprocals[x]
            current_len = len(sums)
            for j in range(current_len):
                new_sum = sums[j] + val
                sums.append(new_sum)
                masks.append(masks[j] | (1 << i))
                
        # Map sum to mask
        seen = {}
        for i, s in enumerate(sums):
            if s in seen:
                # Collision found between masks[i] and seen[s]
                # The difference is the collision
                # We need to extract the coefficients
                # c_k = 1 if bit k in mask1 XOR mask2?
                # Actually, sum(mask1) - sum(mask2) = 0
                # So coefficients are 1 for bits in mask1 not in mask2, -1 for bits in mask2 not in mask1.
                mask1 = masks[i]
                mask2 = seen[s]
                if mask1 == mask2:
                    continue # Should not happen if sums are unique for masks
                # Construct the collision vector
                # We just need to know which elements are involved to potentially remove one.
                # For now, just return the elements involved.
                involved = []
                for k in range(n):
                    bit1 = (mask1 >> k) & 1
                    bit2 = (mask2 >> k) & 1
                    if bit1 != bit2:
                        involved.append(S[k])
                return involved
            else:
                seen[s] = masks[i]
        return None
=======
    def is_dissociated(S):
        """
        Check if all subset sums of reciprocals of S are distinct.
        S is a list of integers.
        Uses integer arithmetic with LCM for speed.
        """
        if not S:
            return True
        
        # Calculate LCM of S
        def lcm(a, b):
            return a // math.gcd(a, b) * b
            
        L = 1
        for x in S:
            L = lcm(L, x)
            
        # Precompute integer weights
        weights = [L // x for x in S]
        
        # Generate all subset sums as integers
        sums = [0]
        for w in weights:
            current_len = len(sums)
            for i in range(current_len):
                sums.append(sums[i] + w)
                
        # Check for duplicates
        if len(sums) != len(set(sums)):
            return False
        return True
>>>>>>> REPLACE
