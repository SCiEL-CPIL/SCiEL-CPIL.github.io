import time

def construct():
        TARGET = 123457
        start_time = time.time()
        DEADLINE = 85.0

        # Primes to check
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

        smooth_set = {1}
        smooth_list = [1]

        for p in primes:
            if time.time() - start_time > DEADLINE:
                # Return best found so far? 
                # We haven't found a pair yet in this loop iteration if we are here.
                # We should return the last valid pair found? 
                # But we only return when we find a pair.
                # If we time out, we should return a valid pair with a known P.
                # The first pair found is the best.
                # If we haven't found one, we are in trouble.
                # But for small primes, it's fast.
                return [1, TARGET - 1]

            new_nums = []
            for s in smooth_list:
                v = s * p
                while v <= TARGET:
                    if v not in smooth_set:
                        smooth_set.add(v)
                        new_nums.append(v)
                    v *= p
            
            smooth_list.extend(new_nums)
            
            # Check for pair
            # Optimization: Only need to check if any a in smooth_set has b in smooth_set
            # We can iterate over the smaller set or just the whole set.
            # Since we just added new_nums, maybe the pair involves a new number?
            # Not necessarily. But checking the whole set is safe.
            # To speed up, we can break early.
            found = False
            for a in smooth_set:
                b = TARGET - a
                if b in smooth_set:
                    # Found pair
                    # Ensure a, b are positive
                    if a > 0 and b > 0:
                        return [a, b]
            
            # If not found, continue to next prime

        return [1, TARGET - 1]
