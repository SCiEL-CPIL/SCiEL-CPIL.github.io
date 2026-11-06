import time
import math
import heapq

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    target = 123457
    
    # Generate k-smooth numbers using a priority queue
    # This is more efficient than the current approach for larger k
    def generate_smooth_numbers(primes, limit):
        """Generate all numbers <= limit whose prime factors are in primes."""
        if not primes:
            return {1}
        
        smooth = set()
        # Start with 1
        pq = [(1, 0)]  # (value, next_prime_index)
        smooth.add(1)
        
        while pq:
            val, idx = heapq.heappop(pq)
            for i in range(idx, len(primes)):
                new_val = val * primes[i]
                if new_val > limit:
                    break
                if new_val not in smooth:
                    smooth.add(new_val)
                    heapq.heappush(pq, (new_val, i))
        
        return smooth
    
    # Try increasing k until we find a solution
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # For each k, generate k-smooth numbers and check if target is a sum of two
    for k in range(1, len(primes_list) + 1):
        if time.time() > deadline - 5:
            break
        
        primes = primes_list[:k]
        smooth_set = generate_smooth_numbers(primes, target)
        
        # Check if target can be written as sum of two k-smooth numbers
        for x in smooth_set:
            y = target - x
            if y in smooth_set and y > 0:
                # Found a solution with max prime factor <= primes_list[k-1]
                # Since we're checking in order of increasing k, this is optimal
                return [x, y]
        
        # If we're getting close to deadline, do a fallback scan
        if time.time() > deadline - 10:
            break
    
    # Fallback: full scan to find best pair
    # This ensures we always return a valid answer
    max_n = target - 1
    
    # Compute largest prime factor for all numbers up to target-1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
                else:
                    lpf[j] = i  # i is the largest prime factor since we go in increasing order
    
    best_max = float('inf')
    best_a = 1
    
    for a in range(1, target):
        if time.time() > deadline - 1:
            break
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
    
    return [best_a, target - best_a]
