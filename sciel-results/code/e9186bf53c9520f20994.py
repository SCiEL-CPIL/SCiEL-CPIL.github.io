# EVOLVE-BLOCK-START
def construct():
    n = 255
    # Prime factors of 255
    primes = [3, 5, 17]
    
    best_prime = None
    max_count = 0
    
    for p in primes:
        count = n // p
        if count > max_count:
            max_count = count
            best_prime = p
            
    # If no prime found (should not happen), fallback to {255}
    if best_prime is None:
        return [255]
        
    result = [i for i in range(1, n + 1) if i % best_prime == 0]
    return result
# EVOLVE-BLOCK-END
