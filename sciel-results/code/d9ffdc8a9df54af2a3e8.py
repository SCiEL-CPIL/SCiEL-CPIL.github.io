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
    num_primes = len(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    def get_vector(n):
        vec = [0] * num_primes
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
        for i in range(n):
            prod_vec = [0] * num_primes
            for j in range(i, n):
                v = vectors[seq[j]]
                for k in range(num_primes):
                    prod_vec[k] += v[k]
                t = tuple(prod_vec)
                if t in seen:
                    return False
                seen.add(t)
        return True
    
    # 4. Incremental validation: check if inserting c at pos into seq is valid
    # Only need to check blocks that include the new element at position pos
    def is_valid_insertion(seq, c, pos):
        # Check strictly increasing
        if pos > 0 and seq[pos-1] >= c:
            return False
        if pos < len(seq) and seq[pos] <= c:
            return False
        
        n = len(seq)
        # Collect all existing block products that don't include position pos
        # and check for collisions with new blocks that include pos
        
        # First, compute all existing block products
        existing_products = set()
        for i in range(n):
            prod_vec = [0] * num_primes
            for j in range(i, n):
                v = vectors[seq[j]]
                for k in range(num_primes):
                    prod_vec[k] += v[k]
                t = tuple(prod_vec)
                # Only add blocks that don't include position pos
                if not (i <= pos-1 and j >= pos):  # Block spans across insertion point
                    existing_products.add(t)
        
        # Now check new blocks that include the inserted element at pos
        c_vec = vectors[c]
        for i in range(pos + 1):  # Start from 0 to pos (in new sequence, c is at pos)
            # In the new sequence, blocks starting at i and ending at j where i <= pos <= j
            # i ranges from 0 to pos, j ranges from pos to n
            prod_vec = [0] * num_primes
            for j in range(i, pos + 1):
                if j < pos:
                    v = vectors[seq[j]]
                else:
                    v = c_vec
                for k in range(num_primes):
                    prod_vec[k] += v[k]
                t = tuple(prod_vec)
                if t in existing_products:
                    return False
                # Also check against other new blocks
                # We'll collect all new block products and check for duplicates
                existing_products.add(t)
        
        return True
    
    # 5. Local search: try to improve sequence by removing and reinserting elements
    def local_search(seq, max_iterations=100):
        if time.time() > deadline:
            return seq
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(seq)
            # Try removing each element and reinserting it in a different position
            for i in range(n):
                if time.time() > deadline:
                    break
                elem = seq[i]
                remaining = seq[:i] + seq[i+1:]
                
                # Try inserting elem at each position in remaining
                for pos in range(len(remaining) + 1):
                    # Check if this gives a different arrangement
                    if pos == i or (pos < i and remaining[pos] == elem):
                        continue
                    # Check strictly increasing
                    if pos > 0 and remaining[pos-1] >= elem:
                        continue
                    if pos < len(remaining) and remaining[pos] <= elem:
                        continue
                    
                    candidate = remaining[:pos] + [elem] + remaining[pos:]
                    if is_valid_sequence(candidate):
                        # This is a valid move; accept it (or could use simulated annealing)
                        seq = candidate
                        improved = True
                        break
                if improved:
                    break
        
        return seq
    
    # 6. Heuristic Search
    best_seq = list(primes)
    best_len = len(primes)
    
    # Pre-sort composites by number of prime factors (ascending) - simpler composites first
    def num_prime_factors(n):
        count = 0
        temp = n
        for p in all_primes:
            while temp % p == 0:
                count += 1
                temp //= p
        return count
    
    composites_sorted = sorted(composites, key=lambda x: (num_prime_factors(x), x))
    
    for trial in range(10000):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        # Start with primes
        current_seq = list(primes)
        remaining_composites = composites_sorted[:]
        
        # Randomize the order but with some structure
        # Mix: some in sorted order, some random
        if trial % 3 == 0:
            random.shuffle(remaining_composites)
        elif trial % 3 == 1:
            # Reverse sorted
            remaining_composites = remaining_composites[::-1]
        else:
            # Partial shuffle
            half = len(remaining_composites) // 2
            first_half = remaining_composites[:half]
            second_half = remaining_composites[half:]
            random.shuffle(first_half)
            random.shuffle(second_half)
            remaining_composites = first_half + second_half
        
        # Try to insert each composite
        for c in remaining_composites:
            if time.time() > deadline:
                break
            
            # Find the correct position to maintain strictly increasing order
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1
            
            # If c is already in the sequence, skip
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
            
            # Check if inserting c at pos maintains unique products
            if is_valid_insertion(current_seq, c, pos):
                current_seq = current_seq[:pos] + [c] + current_seq[pos:]
        
        # Local search to improve
        if len(current_seq) >= best_len and time.time() < deadline - 2.0:
            current_seq = local_search(current_seq)
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq
    
    return best_seq
# EVOLVE-BLOCK-END
