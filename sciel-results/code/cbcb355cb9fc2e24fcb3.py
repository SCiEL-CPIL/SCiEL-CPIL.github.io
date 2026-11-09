import random
import math
import time
from itertools import combinations

def construct():
    # Deadline guard: 90s limit, use 80s for safety
    deadline = time.time() + 80.0
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # Fallback from seed

    def check_pairwise_coprime(nums):
        """Check if all distinct numbers in the list are pairwise coprime."""
        distinct_nums = list(set(nums))
        n = len(distinct_nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct_nums[i], distinct_nums[j]) != 1:
                    return False
        return True

    def get_distinct_sum_count(A, B):
        """Calculate the number of distinct sums in A+B."""
        return len({a + b for a in A for b in B})

    # Precompute primes up to 256 (max sum 128+128)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, val in enumerate(is_prime) if val]

    primes = sieve(256)
    prime_set = set(primes)

    # Strategy 1: Structured search for 9 distinct prime sums
    # If all 9 sums are distinct primes, they're automatically pairwise coprime.
    # We want a_i + b_j to be prime for all i,j.
    # Key insight: if we pick A with all odd and B with all even (or vice versa),
    # all sums are odd, which are candidates for primes (except 2).
    
    # Let's try: A = {odd1, odd2, odd3}, B = {even1, even2, even3}
    # All sums will be odd. We need all 9 to be distinct primes.
    
    # Precompute for each odd number a in [1,128], which even numbers b in [1,128]
    # make a+b prime.
    odd_numbers = [x for x in range(1, 129) if x % 2 == 1]
    even_numbers = [x for x in range(1, 129) if x % 2 == 0]
    
    # For each pair (a, b) where a is odd, b is even, check if a+b is prime
    valid_pairs = {}  # a -> list of b's where a+b is prime
    for a in odd_numbers:
        valid_pairs[a] = []
        for b in even_numbers:
            if (a + b) in prime_set:
                valid_pairs[a].append(b)
    
    # Now search for three odd numbers a1, a2, a3 and three even numbers b1, b2, b3
    # such that all 9 sums are distinct primes.
    # To speed up, we can use backtracking with pruning.
    
    def find_prime_solution():
        """Search for A (3 odds) and B (3 evens) with all 9 sums distinct primes."""
        # Get odd numbers that have at least 3 valid even partners
        candidate_odds = [a for a in odd_numbers if len(valid_pairs[a]) >= 3]
        
        if len(candidate_odds) < 3:
            return None
        
        # Try combinations of 3 odd numbers
        for a1, a2, a3 in combinations(candidate_odds, 3):
            if time.time() > deadline:
                return None
            
            # For each triple of odds, find common even partners
            common_evens = set(valid_pairs[a1]) & set(valid_pairs[a2]) & set(valid_pairs[a3])
            common_evens_list = list(common_evens)
            
            if len(common_evens_list) < 3:
                continue
            
            # Try combinations of 3 even numbers from common partners
            for b1, b2, b3 in combinations(common_evens_list, 3):
                if time.time() > deadline:
                    return None
                
                A = [a1, a2, a3]
                B = [b1, b2, b3]
                
                sums = [a + b for a in A for b in B]
                
                # Check if all 9 sums are distinct
                if len(set(sums)) != 9:
                    continue
                
                # Check if all are primes (they should be by construction, but verify)
                if all(s in prime_set for s in sums):
                    return A, B
        
        return None
    
    # Try structured search first
    result = find_prime_solution()
    if result is not None:
        return result
    
    # Strategy 2: Randomized search with bias toward prime-friendly numbers
    # Mix of strategies: sometimes try all-odd/all-even, sometimes random
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Choose strategy: 70% structured (odd/even), 30% fully random
        if random.random() < 0.7:
            # Structured: A from odds, B from evens (or vice versa)
            if random.random() < 0.5:
                A = random.sample(odd_numbers, 3)
                B = random.sample(even_numbers, 3)
            else:
                A = random.sample(even_numbers, 3)
                B = random.sample(odd_numbers, 3)
        else:
            # Fully random
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
        
        # Calculate sums
        sums = [a + b for a in A for b in B]
        
        # Check if all 9 sums are distinct
        if len(set(sums)) != 9:
            continue
        
        # Check if all are primes
        if all(s in prime_set for s in sums):
            return A, B
        
        # If not all primes, check pairwise coprimality for the distinct sums
        distinct_sums = list(set(sums))
        if check_pairwise_coprime(distinct_sums):
            return A, B
    
    # If time runs out or no perfect solution found, return the best found or fallback
    # The fallback [[1, 7, 13], [4, 10, 16]] has sums:
    # 1+4=5, 1+10=11, 1+16=17
    # 7+4=11, 7+10=17, 7+16=23
    # 13+4=17, 13+10=23, 13+16=29
    # Distinct sums: {5, 11, 17, 23, 29}. Count = 5.
    # Are they pairwise coprime?
    # 5, 11, 17, 23, 29 are all primes. Yes.
    # So fallback is valid with score corresponding to 5.
    
    return best_result
