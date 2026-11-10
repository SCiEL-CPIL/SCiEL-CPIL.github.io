from fractions import Fraction

def construct():
    selected = []
    # Set of all possible subset sums of reciprocals of selected elements
    # Start with 0 (empty sum)
    sums = {Fraction(0, 1)}
    
    # Iterate from largest to smallest
    for cand in range(24, 0, -1):
        recip = Fraction(1, cand)
        
        # Check if recip is already a subset sum of existing elements
        # If so, adding cand would make 1/cand = sum of others, which is invalid
        if recip not in sums:
            # It is safe to add cand
            # Update the set of sums
            # New sums are old sums + recip
            # We must be careful to use the old sums for generation
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + recip)
            sums = new_sums
            selected.append(cand)
            
    # Ensure nonempty
    if not selected:
        return [1]
        
    return list(selected)
