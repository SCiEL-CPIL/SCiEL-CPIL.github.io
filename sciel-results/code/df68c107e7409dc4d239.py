def construct():
        import math
        
        n = 512
        best_set = []
        best_size = 0
        
        # Find M <= 512 with maximum number of divisors
        # And verify that the set of divisors is valid (it always is if M <= 512)
        
        def count_divisors(x):
            count = 0
            for i in range(1, int(math.sqrt(x)) + 1):
                if x % i == 0:
                    count += 1
                    if i != x // i:
                        count += 1
            return count
        
        max_divs = 0
        best_M = 1
        for m in range(1, n + 1):
            d = count_divisors(m)
            if d > max_divs:
                max_divs = d
                best_M = m
        
        # Construct the set of divisors of best_M
        divisors = []
        for i in range(1, int(math.sqrt(best_M)) + 1):
            if best_M % i == 0:
                divisors.append(i)
                if i != best_M // i:
                    divisors.append(best_M // i)
        
        return divisors
