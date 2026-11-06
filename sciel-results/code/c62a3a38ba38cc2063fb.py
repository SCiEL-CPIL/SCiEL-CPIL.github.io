import random
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55s to be safe
    
    # 1. Identify primes and composites in [2, 128]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]
    
    # 2. Precompute factorization vectors
    # We need primes up to 128
    all_primes = primes
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    def get_vector(n):
        vec = [0] * len(all_primes)
        temp = n
        for i, p in enumerate(all_primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        return tuple(vec)
        
    vectors = {x: get_vector(x) for x in range(2, 129)}
    
    # 3. Helper to check validity of a strictly increasing sequence
    def is_valid_sequence(seq):
        if not seq:
            return True
        # Check strictly increasing
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        
        seen = set()
        n = len(seq)
        # Optimization: Iterate starting index from right to left? 
        # Or just standard. 
        # We can stop early if we find a collision.
        for i in range(n):
            prod_vec = [0] * len(all_primes)
            for j in range(i, n):
                v = vectors[seq[j]]
                for k in range(len(all_primes)):
                    prod_vec[k] += v[k]
                t = tuple(prod_vec)
                if t in seen:
                    return False
                seen.add(t)
        return True

    # 4. Heuristic Search
    best_seq = list(primes)
    best_len = len(primes)
    
    num_trials = 5000 # Increased trials due to simpler insertion logic
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        # Start with primes, shuffled? No, primes must be in increasing order for the base.
        # But we can start with a random subset or just primes.
        # Let's start with primes in increasing order.
        current_seq = list(primes)
        remaining_composites = composites[:]
        random.shuffle(remaining_composites)
        
        # Try to insert each composite
        for c in remaining_composites:
            # Find the correct position to maintain strictly increasing order
            # Since current_seq is sorted, we can use binary search or linear scan.
            # Linear scan is fine for small N.
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1
            
            # If c is already in the sequence (shouldn't happen as we pick from remaining), skip
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
                
            # Check if inserting c at pos maintains unique products
            # We only need to check the validity of the new sequence.
            # Optimization: We can check if the insertion causes any collision.
            # A full re-check is O(N^2 * P). N is small (< 128).
            candidate = current_seq[:pos] + [c] + current_seq[pos:]
            
            if is_valid_sequence(candidate):
                current_seq = candidate
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq
            
    return best_seq
# EVOLVE-BLOCK-END
