import random
import math
import time

def construct():
    # Set a time limit to ensure we finish within 60s
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer
    
    # Precompute LCMs or just calculate on the fly?
    # Calculating LCM on the fly is fine for small sets.
    
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)
    
    def is_valid_subset(subset):
        if not subset:
            return True
        # Calculate LCM of the subset
        D = 1
        for x in subset:
            D = get_lcm(D, x)
        
        # Generate all subset sums
        sums = {0}
        for x in subset:
            inc = D // x
            new_sums = set()
            for s in sums:
                new_sums.add(s + inc)
            # Check for collision with existing sums
            if not new_sums.isdisjoint(sums):
                return False
            sums.update(new_sums)
        return True

    # We want to find the largest valid subset.
    # Try sizes from 22 down to 1.
    # For each size, use randomized backtracking.
    
    all_elements = list(range(1, 23))
    
    # Heuristic: Sort elements in descending order for the backtracking tree?
    # Larger numbers have smaller reciprocals, so they are "less likely" to cause collisions
    # with each other compared to small numbers.
    # However, we are choosing a subset.
    
    def backtrack(remaining, current, current_sums, current_lcm, target_size, depth):
        if time.time() - start_time > time_limit:
            return None
        
        if len(current) == target_size:
            return current
        
        if not remaining:
            return None
            
        # Heuristic: Try elements in a randomized order?
        # Or try to pick elements that are "far" from the current sum range?
        
        # To avoid deep recursion with little progress, we can prune if:
        # Even if we add all remaining elements, we can't reach target_size
        if len(current) + len(remaining) < target_size:
            return None
            
        # Randomize the order of remaining elements to explore different branches
        # But for backtracking, we need to be systematic to find *a* solution.
        # Randomization helps avoid getting stuck in one bad path.
        
        shuffled_remaining = remaining[:]
        random.shuffle(shuffled_remaining)
        
        for x in shuffled_remaining:
            new_current = current + [x]
            new_lcm = get_lcm(current_lcm, x) if current_lcm else x
            inc = new_lcm // x
            
            # Calculate new sums
            # Optimization: If current_sums is empty, new_sums is {0, inc}
            # Otherwise, new_sums = current_sums U {s + inc for s in current_sums}
            
            # We need to check if {s + inc for s in current_sums} intersects current_sums
            # And also if the new set of sums has internal collisions?
            # If current_sums were distinct, and we add inc, the new set is S U (S+inc).
            # Collisions can occur between:
            # 1. s1 + inc == s2 + inc => s1 == s2 (impossible as S is distinct)
            # 2. s1 + inc == s2 => s1 - s2 == -inc
            # So we just need to check if (S + inc) intersects S.
            
            # Wait, is that sufficient?
            # The new set of sums is $S' = S \cup (S + \text{inc})$.
            # If $S$ had distinct elements, does $S'$ have distinct elements?
            # Only if $S \cap (S + \text{inc}) = \emptyset$.
            # Yes, because any element in $S'$ is either in $S$ or in $S+\text{inc}$.
            # If an element is in both, it's a collision.
            # If an element is in $S$ twice, it was already a collision (but S is valid).
            # If an element is in $S+\text{inc}$ twice, it implies $s_1+\text{inc} = s_2+\text{inc} \implies s_1=s_2$ (impossible).
            
            # So the check is simply: is $(S + \text{inc})$ disjoint from $S$?
            
            # However, we must also ensure that the new sums $S+\text{inc}$ don't collide with each other?
            # No, as shown above.
            
            # So:
            shifted_sums = {s + inc for s in current_sums}
            if shifted_sums.isdisjoint(current_sums):
                new_sums = current_sums.union(shifted_sums)
                result = backtrack(
                    [y for y in remaining if y != x], 
                    new_current, 
                    new_sums, 
                    new_lcm, 
                    target_size, 
                    depth + 1
                )
                if result is not None:
                    return result
            # Else, collision, skip this x
            
        return None

    # Try to find a solution of size K
    # Start with a reasonable guess, e.g., 12, and go up?
    # Or start from 22 and go down?
    # Starting from 22 is safer to guarantee maximality if we find one.
    # But it might take longer if no solution of size 22 exists.
    # Let's try a few random restarts for each size?
    
    # Given the small N, let's try sizes from 15 down to 1?
    # Or just run a randomized local search which is simpler and robust.
    
    # Let's switch to Randomized Local Search (Hill Climbing) as it's easier to parallelize in thought and code.
    # Actually, Backtracking is better for finding the *maximum* if the space is structured.
    
    # Let's stick to Backtracking but with a time limit and random restarts.
    
    best_set = []
    
    # Try target sizes from high to low
    for target_size in range(22, 0, -1):
        found = False
        # Try a few random permutations for this target size
        for _ in range(20): # 20 restarts
            if time.time() - start_time > time_limit:
                return best_set if best_set else [1]
            
            # Shuffle the pool of elements
            pool = all_elements[:]
            random.shuffle(pool)
            
            # Use a simple recursive search with the shuffled pool
            # We need to pass the pool to the backtracking function
            # The previous backtracking function assumed a fixed 'remaining' list.
            # We can modify it to take the pool as the initial 'remaining'.
            
            # Let's re-implement the backtracking to take the pool.
            
            def search(pool, current, current_sums, current_lcm, target_size):
                if time.time() - start_time > time_limit:
                    return None
                if len(current) == target_size:
                    return current
                if len(current) + len(pool) < target_size:
                    return None
                
                # Heuristic: Pick the first element that works?
                # Since pool is shuffled, this explores random branches.
                
                for i in range(len(pool)):
                    x = pool[i]
                    new_pool = pool[:i] + pool[i+1:]
                    
                    new_lcm = get_lcm(current_lcm, x) if current_lcm else x
                    inc = new_lcm // x
                    
                    shifted_sums = {s + inc for s in current_sums}
                    if shifted_sums.isdisjoint(current_sums):
                        new_sums = current_sums.union(shifted_sums)
                        result = search(new_pool, current + [x], new_sums, new_lcm, target_size)
                        if result is not None:
                            return result
                return None

            result = search(pool, [], {0}, 0, target_size)
            if result is not None:
                if len(result) > len(best_set):
                    best_set = result
                found = True
                break # Found a valid set of this size, no need to try smaller sizes
        
        if found:
            break
            
    if not best_set:
        # Fallback: should not happen, but [1] is always valid
        return [1]
        
    return best_set
