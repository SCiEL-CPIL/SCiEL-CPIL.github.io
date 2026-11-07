import time
import random
import sys

START_TIME = time.time()
TIME_LIMIT = 85.0

def subset_sums(nums):
    """Generate all nonempty subset sums of nums."""
    sums = {0}
    for x in nums:
        sums = sums | {s + x for s in sums}
    sums.discard(0)
    return sums

def is_valid_set(nums):
    """Check if a set of numbers satisfies the constraint."""
    if not nums:
        return True
    sums = subset_sums(nums)
    if not sums:
        return True
        
    sum_set = set(sums)
    max_sum = max(sum_set)
    
    # Check divisibility
    # For each sum s, check if any multiple k*s (k>=2) is in sum_set
    for s in sum_set:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def construct():
    """
    Returns a list of distinct integers in 1..256 maximizing size,
    such that no two distinct nonempty subset sums divide one another.
    """
    best_set = []
    best_size = 0
    
    # Pool of candidates
    pool = list(range(1, 257))
    
    # We will run multiple iterations of Local Search
    # Each iteration starts from a random seed or the best found so far
    
    max_iterations = 5000 # Adjust based on time
    
    # Initialize with a small valid set
    # A set of primes in a small range is often a good start
    # Or just random
    
    current_set = []
    
    # Pre-calculate some random orders
    rng = random.Random(42)
    
    for i in range(max_iterations):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Strategy: Start with a random small set, then grow
        if i % 100 == 0:
            # Restart with a random seed set of size 3-5
            current_set = rng.sample(pool, rng.randint(3, 5))
            if not is_valid_set(current_set):
                # Ensure start is valid
                # Greedily fix
                valid_start = []
                for x in current_set:
                    if is_valid_set(valid_start + [x]):
                        valid_start.append(x)
                current_set = valid_start
        else:
            # Continue from previous state (Iterated Local Search)
            # Occasionally perturb
            if rng.random() < 0.1:
                # Perturb: remove 1-2 elements
                for _ in range(rng.randint(1, 2)):
                    if current_set:
                        current_set.pop(rng.randint(0, len(current_set)-1))
        
        # Local Search Step: Try to add, swap, or remove
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            current_set_copy = list(current_set)
            
            # 1. Try to add a random element
            if len(current_set) < 30:
                # Try a few random additions
                for _ in range(5):
                    candidate = rng.choice(pool)
                    if candidate not in current_set:
                        if is_valid_set(current_set + [candidate]):
                            current_set.append(candidate)
                            improved = True
                            break
                            
            # 2. Try to swap an element
            if len(current_set) > 0:
                for _ in range(5):
                    idx = rng.randint(0, len(current_set)-1)
                    old_elem = current_set[idx]
                    candidate = rng.choice(pool)
                    if candidate != old_elem and candidate not in current_set:
                        new_set = current_set[:idx] + [candidate] + current_set[idx+1:]
                        if is_valid_set(new_set):
                            current_set = new_set
                            improved = True
                            break
                            
            # 3. If no improvement, try removing an element to allow future additions
            # This is a "kick"
            if not improved and len(current_set) > 1:
                if rng.random() < 0.5:
                    idx = rng.randint(0, len(current_set)-1)
                    current_set.pop(idx)
                    improved = True # Force another loop iteration
            
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            if best_size >= 30:
                break

    # Final Polish: Try to add any remaining element
    for x in pool:
        if x not in best_set:
            if is_valid_set(best_set + [x]):
                best_set.append(x)
                
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
