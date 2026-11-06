"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Find 12 distinct positive integers with all subset sums distinct, minimizing the largest element.
    """
    start_time = time.time()
    deadline = 80  # seconds, with safety margin before 90s kill
    
    # Greedy construction: build the set element by element
    # At each step, try the smallest candidate that keeps all subset sums distinct
    
    def subset_sums(a):
        """Compute all subset sums of list a."""
        sums = {0}
        for x in a:
            new_sums = {s + x for s in sums}
            sums = sums | new_sums
        return sums
    
    def greedy_construct():
        """Greedy: add smallest possible element at each step."""
        result = []
        current_sums = {0}
        
        for i in range(12):
            # Try candidates starting from 1
            candidate = 1
            while True:
                # Check if adding candidate keeps all subset sums distinct
                new_sums = current_sums | {s + candidate for s in current_sums}
                if len(new_sums) == (1 << (i + 1)):
                    result.append(candidate)
                    current_sums = new_sums
                    break
                candidate += 1
                # Safety: if we've gone too high, break
                if candidate > 1048576:
                    return None
                if time.time() - start_time > deadline:
                    return None
        return result
    
    # Try greedy first
    result = greedy_construct()
    if result is not None:
        return result
    
    # Fallback: powers of 2
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
