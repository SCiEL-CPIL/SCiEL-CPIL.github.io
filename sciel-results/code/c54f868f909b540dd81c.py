"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def compute_phi_block(start, length, primes):
    """
    Compute phi(n) for n in [start, start+length-1] using segmented sieve.
    Handles large prime factors > sqrt(10^12) by tracking remaining factors.
    """
    phi = np.arange(start, start + length, dtype=np.int64)
    rem = np.arange(start, start + length, dtype=np.int64)
    
    for p in primes:
        if p * p > start + length - 1:
            break
        first = ((start + p - 1) // p) * p
        for j in range(first, start + length, p):
            idx = j - start
            phi[idx] = phi[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    # Correct for large prime factors
    for i in range(length):
        if rem[i] > 1:
            phi[i] = phi[i] // rem[i] * (rem[i] - 1)
            
    return phi

def find_run_in_block(phi_vals, start, k, deadline):
    """
    Find a window of size k with distinct phi values in the block.
    Returns the start index of the window if found, else -1.
    """
    if len(phi_vals) < k:
        return -1
        
    # Initialize window counts
    counts = defaultdict(int)
    for i in range(k):
        counts[phi_vals[i]] += 1
        
    # Check initial window
    if max(counts.values()) == 1:
        return start
        
    # Slide window
    for i in range(k, len(phi_vals)):
        # Remove left
        left_val = phi_vals[i - k]
        counts[left_val] -= 1
        if counts[left_val] == 0:
            del counts[left_val]
            
        # Add right
        right_val = phi_vals[i]
        counts[right_val] += 1
        
        # Check if all distinct
        if max(counts.values()) == 1:
            return start + (i - k + 1)
            
        # Time check
        if i % 10000 == 0 and time.time() > deadline:
            return -1
            
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    # We want to find a run of length 200.
    # Scan blocks of size 10^6.
    block_size = 10**6
    k_target = 200
    
    # Start scanning from near 10^12 downwards, or random?
    # Let's try a few random large starts.
    import random
    rng = random.Random(42)
    
    # Generate candidate start positions for blocks
    # We need start + block_size - 1 <= 10^12
    # So start <= 10^12 - 10^6
    
    # Try a few blocks near the top
    base_start = 10**12 - block_size
    candidates = []
    for i in range(10):
        candidates.append(base_start - i * block_size)
        
    # Add some random ones
    for _ in range(5):
        s = rng.randint(1, 10**12 - block_size)
        candidates.append(s)
        
    candidates.sort(reverse=True)
    
    best_result = {'start': 1, 'length': 1}
    
    for s_block in candidates:
        if time.time() > deadline - 10:
            break
            
        phi_vals = compute_phi_block(s_block, block_size, primes)
        
        # Find a run of length 200
        idx = find_run_in_block(phi_vals, s_block, k_target, deadline)
        
        if idx != -1:
            return {'start': idx, 'length': k_target}
            
        # If not found, try smaller k?
        # Given the density, 200 should be found easily.
        # If not, we can try smaller k in the same block or next block.
        # For now, just move to next block.
        
    return best_result

# EVOLVE-BLOCK-END
