"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # The optimal set of integers found by analysis:
    # Prime 2: {2, 4, 16} (exponents 1, 2, 4 -> distinct subset sums 0..7)
    # Prime 3: {3, 9}    (exponents 1, 2   -> distinct subset sums 0..3)
    # Prime 5: {5, 25}   (exponents 1, 2   -> distinct subset sums 0..3)
    # Other primes: {p}   (exponent 1      -> distinct subset sums 0, 1)
    
    optimal_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify validity just in case (though it's a static list)
    # Check distinctness of subset products
    # For performance, we can skip full verification if we trust the derivation,
    # but a quick check is safer.
    
    def check_validity(lst):
        products = set()
        k = len(lst)
        # 2^16 = 65536, which is small enough to enumerate
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= lst[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    if check_validity(optimal_set):
        return optimal_set
    else:
        # Fallback to primes if something went wrong
        return [a for a in range(2, 41) if all(a % d for d in range(2, int(a**0.5) + 1))]
# EVOLVE-BLOCK-END
