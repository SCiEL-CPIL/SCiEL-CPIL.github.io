import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_n = 1000000
    min_n = 10000
    
    # List of primes to check
    # We will check y = 2, 3, 5, 7, 11, 13, 17, 19, 23, ...
    # We stop when we find a valid n.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Function to generate y-smooth numbers up to limit
    def generate_smooth(primes_list, limit):
        smooths = {1}
        current = [1]
        # Using a simple expansion method
        # Actually, a recursive generator or heap is better, but for small limits and few primes,
        # we can just iterate.
        
        # Better: Use a set and expand
        # Start with 1
        # For each prime p, multiply existing numbers by p, p^2, etc if <= limit
        
        # Let's use a list and sort at the end
        # Or a set to avoid duplicates
        smooths = set([1])
        
        # We need to be careful with the order of generation to not miss combinations
        # Standard way:
        # result = [1]
        # for p in primes_list:
        #    new_additions = []
        #    for num in result:
        #        val = num
        #        while val * p <= limit:
        #            val *= p
        #            if val not in smooths:
        #                smooths.add(val)
        #                new_additions.append(val)
        #    result.extend(new_additions)
        #    # This is not quite right because we need to combine p with other primes too.
        
        # Correct algorithm for generating smooth numbers:
        # 1. Start with list [1]
        # 2. For each prime p in primes_list:
        #    For each number x in the current list (snapshot):
        #       val = x * p
        #       while val <= limit:
        #          add val to set
        #          val *= p
        #    Merge new numbers into the main list/set
        
        smooths = set([1])
        # We need to process primes one by one, adding all powers of the current prime
        # to all existing combinations.
        
        # Let's use a list of current smooth numbers
        curr_smooths = [1]
        
        for p in primes_list:
            new_smooths = []
            # For each existing smooth number, generate multiples by p
            for num in curr_smooths:
                val = num
                while val * p <= limit:
                    val *= p
                    if val not in smooths:
                        smooths.add(val)
                        new_smooths.append(val)
            # Extend the current list with new numbers found
            curr_smooths.extend(new_smooths)
            
        return smooths

    # We iterate through possible values (y)
    # y must be a prime.
    for y in primes:
        if time.time() > deadline:
            break
            
        # Get primes <= y
        primes_for_y = [p for p in primes if p <= y]
        
        # Generate smooth numbers
        smooths = generate_smooth(primes_for_y, max_n + 1)
        
        # Convert to sorted list to check for consecutive
        sorted_smooths = sorted(smooths)
        
        # Check for consecutive pairs n, n+1 in range [10000, 1000000]
        # We can use a set for O(1) lookup
        smooth_set = set(sorted_smooths)
        
        found = False
        # Iterate through smooth numbers in the range
        # To be efficient, we can just iterate through the sorted list
        for i in range(len(sorted_smooths) - 1):
            n = sorted_smooths[i]
            if n < min_n:
                continue
            if n > max_n:
                break
                
            if (n + 1) in smooth_set:
                return n
        
        # If no pair found for this y, try next y

    # Fallback: if no smooth pair found (should not happen), use LPF sieve
    # But for y=13, we expect to find one.
    
    # If we reach here, we didn't find a pair for y up to 97?
    # Then we fall back to the LPF method to be safe.
    
    max_val = max_n + 1
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    best_n = min_n
    best_val = float('inf')
    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
        current_val = max(lpf[n], lpf[n+1])
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n
