    def is_valid(S):
        if not S: return True
        # Generate all subset sums
        sums = {Fraction(0)}
        for x in S:
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + Fraction(1, x))
            sums = new_sums
        
        # Check each x
        for x in S:
            # Check if 1/x is in sums of S \ {x}
            # We can just check if 1/x is in sums, but sums includes subsets of S.
            # If 1/x is in sums, it might be formed by x itself (subset {x}).
            # So we need to ensure 1/x is NOT formed by a subset of S \ {x}.
            # This is equivalent to: 1/x is in sums, but NOT formed ONLY by x?
            # Actually, if 1/x is in sums, it is a sum of some T subset of S.
            # If x in T, then 1/x = 1/x + sum(T\{x}). => sum(T\{x}) = 0.
            # Since elements are positive, T\{x} must be empty.
            # So if 1/x is in sums, it is either formed by {x} alone, or by a subset of S\{x}.
            # If it is formed by a subset of S\{x}, it's invalid.
            # If it is formed by {x} alone, it's valid (as long as no other subset forms it).
            # Wait, if 1/x = sum(T) where T subset S\{x}, then invalid.
            # So we just need to check if there EXISTS a subset T of S\{x} such that sum(T) = 1/x.
            
            # Efficient way:
            # Generate sums of S\{x}.
            # This is expensive to do for each x.
            
            # Alternative:
            # Generate all subset sums of S.
            # For each x, count how many subsets sum to 1/x.
            # If any subset NOT containing x sums to 1/x, invalid.
            # Or simpler:
            # Let P be set of subset sums of S.
            # Let Q be set of subset sums of S\{x}.
            # If 1/x in Q, invalid.
            
            pass
        return True
    