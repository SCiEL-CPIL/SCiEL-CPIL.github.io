import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Keep well under 60s
    
    S = []
    sums = {Fraction(0)}
    
    # Iterate from largest to smallest to ensure 1/b > 1/a for all a in S
    for b in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, b)
        
        # Check if target is in current sums
        # If it is, adding b would make 1/b a sum of others, which is invalid.
        if target in sums:
            continue
        
        # If not, we can add b
        S.append(b)
        
        # Update sums with new combinations involving b
        # We must iterate over a copy or snapshot because we are adding to the set
        current_sums = list(sums)
        for s in current_sums:
            sums.add(s + target)
            
    if not S:
        return [24]
        
    return S
