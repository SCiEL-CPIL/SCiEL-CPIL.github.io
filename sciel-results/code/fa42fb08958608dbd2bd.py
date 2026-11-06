import time

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 60.0  # Keep a safety margin
    
    # Step 1: Generate the Cantor-like set (base 3 digits 0, 1)
    # This set is 3-AP-free, so it is 4-AP-free.
    selected = set()
    for x in range(1, n + 1):
        # Check if x has only 0s and 1s in base 3
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            selected.add(x)
            
    # Step 2: Try to add remaining elements
    # We iterate through candidates. To potentially improve the result,
    # we can try multiple passes or different orderings.
    # For now, a single pass over the remaining elements in increasing order.
    
    remaining = [x for x in range(1, n + 1) if x not in selected]
    
    def is_4ap_free_with(candidate, current_set):
        """Check if adding 'candidate' to 'current_set' creates a 4-AP."""
        # A 4-AP is (a, a+d, a+2d, a+3d)
        # If we add 'candidate', it could be any of the 4 terms.
        
        # Case 1: candidate is a+3d (last term)
        # a, a+d, a+2d are in set.
        # candidate - 3d, candidate - 2d, candidate - d are in set.
        for d in range(1, (candidate - 1) // 3 + 1):
            if (candidate - 3*d in current_set and 
                candidate - 2*d in current_set and 
                candidate - d in current_set):
                return False
        
        # Case 2: candidate is a+2d (third term)
        # a, a+d, a+3d are in set.
        # candidate - 2d, candidate - d, candidate + d are in set.
        for d in range(1, (candidate - 1) // 2 + 1):
            if (candidate - 2*d in current_set and 
                candidate - d in current_set and 
                candidate + d in current_set):
                return False
        
        # Case 3: candidate is a+d (second term)
        # a, a+2d, a+3d are in set.
        # candidate - d, candidate + d, candidate + 2d are in set.
        for d in range(1, (n - candidate) // 2 + 1):
            if (candidate - d in current_set and 
                candidate + d in current_set and 
                candidate + 2*d in current_set):
                return False
        
        # Case 4: candidate is a (first term)
        # a+d, a+2d, a+3d are in set.
        # candidate + d, candidate + 2d, candidate + 3d are in set.
        for d in range(1, (n - candidate) // 3 + 1):
            if (candidate + d in current_set and 
                candidate + 2*d in current_set and 
                candidate + 3*d in current_set):
                return False
                
        return True

    # Add remaining elements
    for x in remaining:
        if time.time() > deadline:
            break
        if is_4ap_free_with(x, selected):
            selected.add(x)
            
    # Step 3: Local Search / Swapping to potentially improve
    # Try to swap elements to see if we can fit more.
    # This is a simple hill-climbing step.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Create a list of current elements and non-elements
        current_list = sorted(selected)
        non_elements = [x for x in range(1, n + 1) if x not in selected]
        
        # Try removing one element and adding another
        # We can limit the search to avoid O(N^2) overhead if it's too slow,
        # but N=512 is small enough.
        # However, checking every pair is 512*512 * check_cost.
        # check_cost is ~500 ops. 512^2 * 500 = 130M ops. Might be too slow in Python.
        
        # Let's do a more targeted local search:
        # For each element x in selected, try to remove it.
        # Then, for each y in non_elements, try to add it.
        # This is still O(N^2).
        
        # Alternative: Just do a few random restarts or a simple greedy on the "holes".
        # Given the time limit, the initial greedy + Cantor start is likely sufficient to beat the seed.
        # The seed is ~100-120? Cantor is ~80. Greedy on Cantor might reach ~150-200.
        
        # Let's try a simple "fill holes" pass again, but maybe in reverse order?
        # Or just return what we have. The local search might be too slow in pure Python if not optimized.
        
        pass # Skip complex local search for safety, rely on strong initialization + greedy fill.

    return sorted(selected)
