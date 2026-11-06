    def update_L(old_L):
        # old_L is the current L before the move
        # We assume coverage has been updated for the new residue
        
        # 1. Check if old_L is still valid
        if old_L > 0 and coverage[old_L] < 2:
            # L decreased
            if old_L == 1:
                return 0
            # Find first x in 1..old_L-1 with coverage[x] < 2
            # Slice: coverage[1:old_L] corresponds to x=1..old_L-1
            invalid = np.where(coverage[1:old_L] < 2)[0]
            if len(invalid) == 0:
                # All 1..old_L-1 are valid.
                return old_L - 1
            else:
                # invalid[0] is index in slice. x = invalid[0] + 1.
                # First failure at x. L = x - 1 = invalid[0].
                return int(invalid[0])
        
        # 2. old_L is valid. Try to extend
        new_L = old_L
        # Optimization: Only extend if we have time.
        # The while loop is O(1) per step.
        while new_L < MAX_L:
            if coverage[new_L + 1] >= 2:
                new_L += 1
            else:
                break
        return new_L
