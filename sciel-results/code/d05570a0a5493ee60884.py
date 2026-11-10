"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Find 12 distinct positive integers <= 1048576 with all 4096 subset sums distinct.
    Minimize the largest integer.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 12
    max_val = 1048576
    
    # The seed solution is [1, 2, 4, 8, ..., 2048] with max = 2048
    # We need to find a set with a smaller maximum element.
    
    # Key insight: For a set of n distinct positive integers with all subset sums distinct,
    # the minimum possible maximum element is at least 2^(n-1) - (n-1) in some constructions,
    # but we can do better with careful search.
    
    # Actually, the minimum largest element for n distinct positive integers with distinct subset sums
    # is known to be at least 2^(n-1) for "superincreasing" sequences, but non-superincreasing
    # sequences can have smaller max.
    
    # Let's use a randomized local search / simulated annealing approach.
    
    def check_valid(arr):
        """Check if all subset sums are distinct."""
        sums = set()
        sums.add(0)
        for i, a in enumerate(arr):
            # Add new sums by adding a to existing sums
            new_sums = set()
            for s in sums:
                ns = s + a
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums |= new_sums
        return True
    
    def get_max(arr):
        return max(arr)
    
    # Start with a good initial guess
    # The optimal answer for n=12 is known to be around 2048 or slightly less.
    # Let's try to find a set with max < 2048.
    
    # Strategy: Use simulated annealing or random restarts with local search.
    
    best_max = 2048  # From seed
    best_set = [1 << i for i in range(n)]
    
    # Try to find better solutions
    random.seed(42)
    
    # Local search: start from a valid set and try to reduce elements
    def local_search(start_set, max_iterations=10000):
        """Try to reduce the maximum element while maintaining validity."""
        arr = sorted(start_set[:])
        current_max = max(arr)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Try to reduce the largest element
            # Find a smaller value that maintains distinctness
            idx = arr.index(current_max)
            
            # Try replacing the largest element with a smaller one
            improved = False
            for new_val in range(1, current_max):
                if new_val in arr:
                    continue
                test_arr = arr[:]
                test_arr[idx] = new_val
                test_arr.sort()
                
                if check_valid(test_arr):
                    arr = test_arr
                    current_max = max(arr)
                    improved = True
                    break
            
            if not improved:
                # Try swapping two elements
                if len(arr) >= 2:
                    for i in range(len(arr)):
                        for j in range(i + 1, len(arr)):
                            arr[i], arr[j] = arr[j], arr[i]
                            if check_valid(arr):
                                current_max = max(arr)
                                if current_max < best_max:
                                    best_max = current_max
                                    best_set = arr[:]
                                arr[i], arr[j] = arr[j], arr[i]
                            else:
                                arr[i], arr[j] = arr[j], arr[i]
                break
        
        return current_max, arr
    
    # Try multiple random starts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a random starting set
        # Start with a superincreasing-like sequence and perturb
        arr = []
        for i in range(n):
            # Random value, ensuring distinctness
            while True:
                val = random.randint(1, 2048)
                if val not in arr:
                    arr.append(val)
                    break
        arr.sort()
        
        # Check if valid
        if not check_valid(arr):
            continue
        
        # Local search from this point
        cur_max, final_arr = local_search(arr)
        
        if cur_max < best_max:
            best_max = cur_max
            best_set = final_arr[:]
    
    # Also try a more systematic approach: start from the seed and try to reduce
    arr = [1 << i for i in range(n)]
    arr.sort()
    cur_max = max(arr)
    
    # Try to reduce each element one by one
    for i in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
        
        original = arr[i]
        for new_val in range(1, original):
            if new_val in arr:
                continue
            test_arr = arr[:]
            test_arr[i] = new_val
            test_arr.sort()
            
            if check_valid(test_arr):
                arr = test_arr
                break
    
    if max(arr) < best_max:
        best_max = max(arr)
        best_set = arr[:]
    
    return best_set
# EVOLVE-BLOCK-END
