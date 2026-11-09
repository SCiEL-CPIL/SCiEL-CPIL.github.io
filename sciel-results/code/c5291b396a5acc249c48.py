import time
import bisect

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def _has_unique_block_products(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # 1. Initialize with primes
    primes = [x for x in range(2, 129) if _is_prime(x)]
    seq = primes
    used = set(seq)
    
    # 2. Candidates are composites
    composites = [x for x in range(2, 129) if not _is_prime(x)]
    composites.sort()
    
    # 3. Greedy insertion
    # We loop until no more improvements are found
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # We iterate through composites. 
        # If we insert one, we restart the loop to re-evaluate others, 
        # as the insertion might change the validity of subsequent insertions 
        # (though it only adds constraints, so it's safe to continue, 
        # but restarting ensures we always try the smallest available first 
        # in a consistent state).
        for c in composites:
            if time.time() > deadline:
                break
                
            if c in used:
                continue
                
            # Find insertion position
            k = bisect.bisect_left(seq, c)
            
            # Check if position is valid (strictly increasing)
            # k is the index where c would be inserted.
            # seq[k-1] < c < seq[k]
            valid_pos = True
            if k > 0:
                if seq[k-1] >= c:
                    valid_pos = False
            if k < len(seq):
                if seq[k] <= c:
                    valid_pos = False
            
            if not valid_pos:
                continue
                
            # Create new sequence
            new_seq = seq[:k] + [c] + seq[k:]
            
            # Check uniqueness
            if _has_unique_block_products(new_seq):
                seq = new_seq
                used.add(c)
                improved = True
                break # Restart the while loop
        
    return seq
