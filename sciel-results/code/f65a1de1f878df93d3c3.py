"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = []
    
    for p in primes:
        # Get powers of p <= 40
        powers = []
        k = 1
        val = p
        while val <= 40:
            powers.append((val, k))
            k += 1
            val *= p
            
        exponents = [k for val, k in powers]
        
        # Find largest subset of exponents with distinct subset sums
        # Max len is small (<= 5 for p=2)
        best_subset = []
        
        # Sort exponents to help finding large subsets? 
        # Actually, we just need to check all subsets of the available exponents.
        # We want the max size.
        max_size = len(exponents)
        
        # Try sizes from max down to 1
        for size in range(max_size, 0, -1):
            found = False
            # Iterate over combinations of indices
            for combo in itertools.combinations(range(max_size), size):
                subset_exps = [exponents[i] for i in combo]
                # Check if subset sums are distinct
                sums = set()
                valid = True
                # Generate all subset sums of subset_exps
                # There are 2^size sums
                for mask in range(1 << size):
                    s = 0
                    for i in range(size):
                        if mask & (1 << i):
                            s += subset_exps[i]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                
                if valid:
                    # Add corresponding values
                    for i in combo:
                        result.append(powers[i][0])
                    found = True
                    break
            if found:
                break
            if time.time() > deadline:
                # Fallback: just take the prime itself if we run out of time
                # But this is very fast, so unlikely
                result.append(p)
                break
        
        if time.time() > deadline:
            break

    # Ensure we return a list of ints
    return list(result)

# EVOLVE-BLOCK-END
