# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    Uses a greedy approach to select the smallest possible integer at each step
    that preserves the distinctness of all subset sums.
    """
    n = 12
    elements = []
    current_sums = {0}
    candidate = 1
    
    for _ in range(n):
        # Find the smallest integer >= candidate that doesn't cause collisions
        while True:
            # Check if adding 'candidate' creates any duplicate sums
            # New sums would be {candidate + s for s in current_sums}
            # We need this set to be disjoint from current_sums
            # And 'candidate' must not be in 'elements' (distinct integers)
            
            # Optimization: If candidate is in elements, skip
            if candidate in elements:
                candidate += 1
                continue
                
            # Check for sum collisions
            # If any (candidate + s) is in current_sums, there is a collision
            # Note: Since all s >= 0 and candidate > 0, candidate + s > s.
            # So we only need to check if candidate + s is in current_sums.
            # Actually, we just need to check if the set {candidate + s} intersects current_sums.
            
            collision = False
            for s in current_sums:
                if (candidate + s) in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Accept candidate
                elements.append(candidate)
                # Update current_sums
                new_sums = {candidate + s for s in current_sums}
                current_sums.update(new_sums)
                candidate += 1
                break
            else:
                candidate += 1
                
    return elements
# EVOLVE-BLOCK-END
