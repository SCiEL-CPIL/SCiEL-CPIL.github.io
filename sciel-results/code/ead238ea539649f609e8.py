import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def generate_smooth_numbers(limit, B):
        ps = [p for p in primes if p <= B]
        if not ps:
            return {1}
            
        smooths = set()
        # Using a list to build up, avoiding recursion overhead
        # Start with [1]
        current_set = [1]
        
        for p in ps:
            next_set = []
            for x in current_set:
                val = x
                while val <= limit:
                    next_set.append(val)
                    val *= p
            # Merge and deduplicate? 
            # Actually, the standard way is:
            # S_{k} = { n * p_k^j | n in S_{k-1}, j >= 0, n * p_k^j <= limit }
            # But this generates duplicates if not careful? 
            # No, if we process primes one by one, each number has a unique prime factorization.
            # So if we start with {1} and process 2, we get powers of 2.
            # Then process 3, we get 2^i * 3^j.
            # This generates each number exactly once.
            
            # Let's re-implement cleanly
            pass

        # Better approach:
        # Initialize result with 1
        result = {1}
        # For each prime p <= B:
        #   For each x in current result (snapshot):
        #      Multiply x by p, p^2, ... as long as <= limit
        #      Add to new result
        
        # Actually, we can just do:
        curr = [1]
        for p in ps:
            new_vals = []
            for x in curr:
                v = x
                while v <= limit:
                    new_vals.append(v)
                    v *= p
            # The new set of smooth numbers is the union of old ones and new ones?
            # No. 
            # If we have primes p1, p2.
            # Start: {1}
            # Process p1: {1, p1, p1^2, ...}
            # Process p2: For each x in {1, p1, ...}, add x*p2, x*p2^2...
            # The set becomes { p1^i * p2^j ... }
            # So we can just update the list.
            
            # Let's use a set to avoid duplicates if any logic error, but unique factorization ensures no duplicates.
            # However, the list `curr` grows.
            
            pass

        # Let's use the recursive generator, it's clean and fast enough for this size.
        smooths = set()
        def gen(idx, val):
            if val > limit:
                return
            smooths.add(val)
            if idx == len(ps):
                return
            p = ps[idx]
            gen(idx + 1, val)
            v = val
            while v * p <= limit:
                v *= p
                gen(idx + 1, v)
                
        gen(0, 1)
        return smooths

    for B in range(1, 47):
        if time.time() > deadline:
            break
        S = generate_smooth_numbers(target, B)
        for a in S:
            b = target - a
            if b in S:
                return [a, b]
                
    return [1, 123456]
