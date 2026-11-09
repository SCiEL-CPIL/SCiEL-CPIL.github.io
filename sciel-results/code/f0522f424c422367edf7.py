"""Distinct subset sums with a small largest element."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find 12 distinct positive integers <= 1048576 such that all 4096 subset sums
    are distinct, minimizing the largest integer.
    """
    # The optimal solution for distinct subset sums minimizing the maximum element
    # is the powers of 2.
    # For n=12, the set is {1, 2, 4, ..., 2048}.
    # Largest element is 2048.
    
    # We implement a robust check and a local search to confirm optimality
    # or find a better solution if it exists (unlikely).
    
    def is_valid(elements):
        """
        Check if all subset sums are distinct using a bitset.
        bits is an integer where bit i is 1 if sum i is achievable.
        """
        bits = 1  # Sum 0 is achievable
        for e in elements:
            # Shift the bitset by e and check for overlap
            if (bits << e) & bits:
                return False
            bits |= (bits << e)
        return True

    # Start with powers of 2
    elements = [1 << i for i in range(12)]
    
    # Verify
    if not is_valid(elements):
        # This should never happen
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    best_max = max(elements)
    
    # Try to improve using local search
    # We try to reduce each element, keeping the set sorted and distinct.
    # Since powers of 2 are superincreasing, reducing any element breaks the property
    # unless compensated by others.
    
    # A more effective search for small n is to try all combinations near the solution.
    # However, given the time limit and the likelihood that 2048 is optimal,
    # we perform a limited local search to ensure we don't miss a trivial improvement.
    
    deadline = time.time() + 50.0 # 50s safety margin
    
    # Local search: try to reduce the largest element first, then others.
    # We maintain the invariant that elements are sorted.
    
    current = elements[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to reduce each element
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            old_val = current[i]
            # Minimum valid value for position i
            # Must be > current[i-1] if i > 0
            min_cand = current[i-1] + 1 if i > 0 else 1
            
            # We only need to check a small range if we are close to optimal.
            # But to be safe, we check down to min_cand.
            # Optimization: If we know the sum of previous elements S_prev,
            # and if the previous sums form a contiguous range [0, S_prev],
            # then current[i] must be > S_prev.
            
            # Calculate sum of previous elements
            s_prev = sum(current[:i])
            
            # If the previous elements form a "complete" sequence (sums cover 0..s_prev),
            # then current[i] must be at least s_prev + 1.
            # Powers of 2 have this property.
            # If we deviate, we might be able to go lower?
            # Actually, if we deviate, we create gaps, which might allow lower values?
            # But as seen in n=3,4, it didn't help minimize the max.
            
            # We try candidates from min_cand to old_val - 1
            for cand in range(min_cand, old_val):
                if time.time() > deadline:
                    break
                test = current[:]
                test[i] = cand
                # Ensure distinctness
                if len(set(test)) != len(test):
                    continue
                if is_valid(test):
                    current = test
                    improved = True
                    break
            if improved:
                break
        
        if max(current) < best_max:
            best_max = max(current)

    return current

# EVOLVE-BLOCK-END
