import math
import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    # Split primes into small (<= 13) and large (> 13)
    # Primes <= 13 are candidates for merging
    small_primes = [p for p in primes if p <= 13]
    large_primes = [p for p in primes if p > 13]
    
    # Base sum from large primes (they stay as singletons)
    # We will add 1 separately
    base_sum = sum(large_primes)
    
    # Function to get max number <= N with prime factors in G
    def max_num_for_group(G):
        if not G:
            return 1 # This case shouldn't happen for non-empty partition
        max_val = 1
        # Generate all combinations of powers
        # Since G is small, we can use recursion or nested loops
        # Max power of 2 is 7, 3 is 5, 5 is 3, 7 is 2, 11 is 1, 13 is 1
        # Product of max powers is huge, but we bound by 256
        
        def dfs(idx, current_val):
            nonlocal max_val
            if current_val > N:
                return
            max_val = max(max_val, current_val)
            if idx == len(G):
                return
            p = G[idx]
            # Try powers of p
            power = 1
            while current_val * power <= N:
                dfs(idx + 1, current_val * power)
                power *= p
        
        dfs(0, 1)
        return max_val

    # Enumerate all partitions of small_primes
    # small_primes = [2, 3, 5, 7, 11, 13]
    # We can use a simple backtracking to generate set partitions
    
    best_partition_sum = 0
    best_partition = None
    
    n_sp = len(small_primes)
    
    def get_partitions(idx, current_partition):
        nonlocal best_partition_sum, best_partition
        if idx == n_sp:
            # Calculate sum for this partition
            total = 0
            for group in current_partition:
                total += max_num_for_group(group)
            if total > best_partition_sum:
                best_partition_sum = total
                best_partition = [g[:] for g in current_partition]
            return
        
        # Option 1: Add small_primes[idx] to an existing group
        for i in range(len(current_partition)):
            current_partition[i].append(small_primes[idx])
            get_partitions(idx + 1, current_partition)
            current_partition[i].pop()
        
        # Option 2: Start a new group
        current_partition.append([small_primes[idx]])
        get_partitions(idx + 1, current_partition)
        current_partition.pop()

    get_partitions(0, [])
    
    # Construct result
    result = [1]
    for group in best_partition:
        result.append(max_num_for_group(group))
    result.extend(large_primes)
    
    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback to simple prime powers if something went wrong
                return _fallback_construct(N, deadline - time.time())
                
    return [int(r) for r in result]

def _fallback_construct(N, time_budget):
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    return [int(r) for r in result]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}, Count: {len(res)}")
