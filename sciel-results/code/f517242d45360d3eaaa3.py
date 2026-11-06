# EVOLVE-BLOCK-START
import time

def construct():
    n = 12
    max_val = 1048576
    
    # Deadline guard (90s limit, use 85s margin)
    start_time = time.time()
    deadline = start_time + 85
    
    result = []
    current_sum = 0
    
    for i in range(n):
        if time.time() > deadline:
            # In case of timeout, return best effort (though this loop is O(12))
            # For n=12, this is instantaneous.
            break
            
        # The next number must be at least current_sum + 1 to ensure 
        # superincreasing property and thus distinct subset sums.
        next_val = current_sum + 1
        
        # Ensure we stay within the bound (though for n=12 this is not an issue)
        if next_val > max_val:
            # Fallback or error handling, though theoretically unreachable for n=12
            # with this optimal construction
            raise ValueError("Exceeded maximum value bound")
            
        result.append(next_val)
        current_sum += next_val
        
    # If we broke early due to deadline, we might have incomplete result.
    # But for n=12, it will never happen.
    # To be safe, if result is not length 12, we could try to fill or just return.
    # Given the problem constraints, 12 is small enough that time is not a factor.
    
    return result
# EVOLVE-BLOCK-END
