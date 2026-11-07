"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for each number 1..256
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        p = 2
        while p * p <= x:
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
            p += 1
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    # Precompute compatibility using bitmasks for speed
    small_primes = [2, 3, 5, 7, 11, 13]
    small_prime_index = {p: i for i, p in enumerate(small_primes)}
    
    small_mask = [0] * (N + 1)
    large_prime_factor = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        lp = 0
        for p in prime_factors[i]:
            if p in small_prime_index:
                mask |= (1 << small_prime_index[p])
            else:
                lp = p
        small_mask[i] = mask
        large_prime_factor[i] = lp
    
    # Precompute coprime compatibility matrix as a list of sets
    # compatible[i] = set of j such that gcd(i,j)==1 and j != i
    compatible = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if small_mask[i] & small_mask[j]:
                continue
            if large_prime_factor[i] != 0 and large_prime_factor[i] == large_prime_factor[j]:
                continue
            compatible[i].add(j)
            compatible[j].add(i)
    
    def are_coprime(i, j):
        if i == j:
            return False
        return j in compatible[i]
    
    # Get all primes up to 256
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, N + 1) if is_prime(x)]
    
    # Helper: check if num is coprime to all elements in set s
    def can_add_to(num, s):
        """Check if num can be added to set s"""
        if num in s:
            return False
        # Check against all elements in s
        for x in s:
            if x not in compatible[num]:
                return False
        return True
    
    def try_swap_one(old, new, s):
        """Try to swap old for new in s. Returns (success, new_set)"""
        if new in s or new == old or new > N:
            return False, s
        # Check if new is coprime to all other elements
        for x in s:
            if x == old:
                continue
            if x not in compatible[new]:
                return False, s
        return True, (s - {old}) | {new}
    
    def try_swap_two_for_one(old1, old2, new, s):
        """Try to replace old1 and old2 with new in s. Returns (success, new_set)"""
        if new in s or new == old1 or new == old2 or new > N:
            return False, s
        # Check if new is coprime to all other elements
        for x in s:
            if x == old1 or x == old2:
                continue
            if x not in compatible[new]:
                return False, s
        return True, (s - {old1, old2}) | {new}
    
    def try_swap_one_for_two(old, new1, new2, s):
        """Try to replace old with new1 and new2 in s. Returns (success, new_set)"""
        if new1 in s or new2 in s or new1 == new2 or new1 == old or new2 == old:
            return False, s
        if new1 > N or new2 > N:
            return False, s
        # Check if new1 and new2 are coprime to each other
        if new2 not in compatible[new1]:
            return False, s
        # Check if new1 and new2 are coprime to all other elements
        for x in s:
            if x == old:
                continue
            if x not in compatible[new1] or x not in compatible[new2]:
                return False, s
        return True, (s - {old}) | {new1, new2}
    
    # Greedy initialization: start with 1 + all primes, then try to add composites
    current_set = set([1] + primes)
    current_sum = 1 + sum(primes)
    best_set = set(current_set)
    best_sum = current_sum
    
    # Phase 1: Greedy addition - try to add composites in descending order
    all_numbers_desc = list(range(N, 0, -1))
    
    for _ in range(10):
        if time.time() > deadline:
            break
        progress = False
        for num in all_numbers_desc:
            if time.time() > deadline:
                break
            if num in current_set:
                continue
            if can_add_to(num, current_set):
                current_set.add(num)
                current_sum += num
                progress = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = set(current_set)
        if not progress:
            break
    
    # Phase 2: Simulated annealing with multiple move types
    random.seed(42)
    temperature = 200.0
    cooling = 0.9995
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        elements = list(current_set)
        if not elements:
            break
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.5:
            # 1-for-1 swap
            old = random.choice(elements)
            candidates = [x for x in range(2, N + 1) if x not in current_set]
            if not candidates:
                continue
            new = random.choice(candidates)
            
            success, new_set = try_swap_one(old, new, current_set)
            if success:
                delta = new_sum - current_sum if False else (sum(new_set) - current_sum)
                if delta >= 0 or random.random() < math.exp(max(delta, -50) / temperature):
                    current_set = new_set
                    current_sum = sum(current_set)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
        
        elif move_type < 0.8:
            # 2-for-1 swap (replace two elements with one larger one)
            if len(elements) < 2:
                continue
            old1, old2 = random.sample(elements, 2)
            candidates = [x for x in range(2, N + 1) if x not in current_set and x > max(old1, old2)]
            if not candidates:
                continue
            new = random.choice(candidates)
            
            success, new_set = try_swap_two_for_one(old1, old2, new, current_set)
            if success:
                delta = sum(new_set) - current_sum
                if delta >= 0 or random.random() < math.exp(max(delta, -50) / temperature):
                    current_set = new_set
                    current_sum = sum(current_set)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
        
        else:
            # 1-for-2 swap (replace one element with two smaller ones)
            old = random.choice(elements)
            # Find pairs of candidates that sum to more than old
            candidates = [x for x in range(2, N + 1) if x not in current_set]
            if len(candidates) < 2:
                continue
            # Sample a few random pairs
            for _ in range(5):
                new1, new2 = random.sample(candidates, 2)
                if new1 + new2 <= old:
                    continue
                success, new_set = try_swap_one_for_two(old, new1, new2, current_set)
                if success:
                    delta = sum(new_set) - current_sum
                    if delta >= 0 or random.random() < math.exp(max(delta, -50) / temperature):
                        current_set = new_set
                        current_sum = sum(current_set)
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
                        break
        
        temperature *= cooling
    
    # Phase 3: Exhaustive local search on best_set
    if time.time() < deadline - 5:
        for iteration in range(200):
            if time.time() > deadline:
                break
            progress = False
            elements = list(best_set)
            
            # Try all 1-for-1 swaps
            for old in elements:
                if time.time() > deadline:
                    break
                for new in range(2, N + 1):
                    if new in best_set or new == old:
                        continue
                    success, new_set = try_swap_one(old, new, best_set)
                    if success:
                        new_sum = sum(new_set)
                        if new_sum > best_sum:
                            best_set = new_set
                            best_sum = new_sum
                            progress = True
                            break
                if progress:
                    break
            
            # Try all 2-for-1 swaps
            if not progress and len(elements) >= 2:
                for i in range(len(elements)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(elements)):
                        if time.time() > deadline:
                            break
                        old1, old2 = elements[i], elements[j]
                        for new in range(max(old1, old2) + 1, N + 1):
                            if new in best_set:
                                continue
                            success, new_set = try_swap_two_for_one(old1, old2, new, best_set)
                            if success:
                                new_sum = sum(new_set)
                                if new_sum > best_sum:
                                    best_set = new_set
                                    best_sum = new_sum
                                    progress = True
                                    break
                        if progress:
                            break
                    if progress:
                        break
            
            # Try all 1-for-2 swaps
            if not progress:
                for old in elements:
                    if time.time() > deadline:
                        break
                    candidates = [x for x in range(2, N + 1) if x not in best_set]
                    for i in range(len(candidates)):
                        if time.time() > deadline:
                            break
                        for j in range(i + 1, len(candidates)):
                            new1, new2 = candidates[i], candidates[j]
                            if new1 + new2 <= old:
                                continue
                            success, new_set = try_swap_one_for_two(old, new1, new2, best_set)
                            if success:
                                new_sum = sum(new_set)
                                if new_sum > best_sum:
                                    best_set = new_set
                                    best_sum = new_sum
                                    progress = True
                                    break
                        if progress:
                            break
                    if progress:
                        break
            
            if not progress:
                break
    
    # Phase 4: Try adding 1 if not present
    if 1 not in best_set:
        if can_add_to(1, best_set):
            best_set.add(1)
            best_sum += 1
    
    # Final verification
    result = list(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if not are_coprime(result[i], result[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to primes
        result = [1] + primes
    
    # Ensure distinct, in range, correct types
    result = list(set(result))
    result = [int(x) for x in result if 1 <= x <= N]
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
